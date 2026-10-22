#pragma once

// A 64x64x16 register-tiled GEMM with a 4x4 thread tile. The 4x4 tile is the widest
// shape that stays inside the 65,536 registers-per-block limit -- the 128x128 kernel's 8x8
// tile throws at launch for complex<double>. Two kernels: the NN one (double;
// register_wide_transposed.hh serves its other forms) and wide v2 below, every form for float,
// complex<float> and complex<double>.
// evidence: docs/perf/gemm.md#the-wide-scalar-kernel

#include <src/sycl/device_scalar.hh>
#include <src/ops/gemm/gemm_kernels.hh>
#include <src/ops/gemm/kernels/copy_async.hh>
#include <src/ops/gemm/kernels/vec4.hh>
#include <src/ops/gemm/kernels/register_tiled_common.hh>

#include <src/linalg-impl.hh>
#include <src/sycl/kernel_attrs.hh>
#include <src/sycl/local_mem.hh>

#include <batchlas/backend_config.h>
#include <batchlas/settings.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace batchlas::sycl_gemm {

namespace wide_scalar {

// std::complex is deliberately kept out of device code; see ../device_scalar.hh.
using batchlas::sycl_device::Cx;
using batchlas::sycl_device::DevMap;
using batchlas::sycl_device::dev_is_zero;
using batchlas::sycl_device::fma_acc;

// beta = 0 is alpha * acc exactly (+ 0 * 0 would turn -0 into +0); callers pass prior = +0 there. The
// beta terms are then -0 (x + -0 and x - +0 are x): loop-invariant selects, no per-element one
// (that cost 2-3%). Complex takes beta.im as +0 in the real part and -0 in the imaginary one.
// evidence: docs/perf/gemm.md#gemm-wp-x-review-beta-zero-keeps-negative-zero
inline float lin_epi(float alpha, float beta, float acc, float prior) {
    return alpha * acc + (beta == 0.f ? -0.f : beta) * prior;
}
inline double lin_epi(double alpha, double beta, double acc, double prior) {
    return alpha * acc + (beta == 0.0 ? -0.0 : beta) * prior;
}
template <typename R>
inline Cx<R> lin_epi(Cx<R> alpha, Cx<R> beta, Cx<R> acc, Cx<R> prior) {
    const bool bz = beta.re == R(0) && beta.im == R(0);
    const R br = bz ? R(-0.0) : beta.re, bi_re = bz ? R(0) : beta.im, bi_im = bz ? R(-0.0) : beta.im;
    Cx<R> o;
    o.re = alpha.re * acc.re - alpha.im * acc.im + br * prior.re - bi_re * prior.im;
    o.im = alpha.re * acc.im + alpha.im * acc.re + br * prior.im + bi_im * prior.re;
    return o;
}

// The access granule is 16 BYTES, not 4 elements, whatever the scalar width:
// that is what keeps an 8-lane LDS phase on exactly the 32 banks.
// may_alias is load-bearing: without it -O3 may reorder the shared stores
// against the fragment loads across the barrier.
template <typename D>
struct alignas(16) __attribute__((may_alias)) Vec16 {
    static constexpr int N = 16 / static_cast<int>(sizeof(D));
    D v[N];
};

template <typename D>
inline const Vec16<D>& vec_ref(const D* p) {
    return *reinterpret_cast<const Vec16<D>*>(p);
}
template <typename D>
inline Vec16<D>& vec_ref(D* p) {
    return *reinterpret_cast<Vec16<D>*>(p);
}

// A raw 16-byte move for staging. It must be a native vector type, not a
// `struct { T v[N]; }`: SROA splits a struct copy back into element loads and
// stores and the 16-byte form is lost. Metal has neither vectors nor double; its emitter
// maps an i128 to a uint4, so there the move is one 16-byte integer.
#if BATCHLAS_DEVICE_NO_VECTOR_TYPES
typedef unsigned __int128 Raw16 __attribute__((may_alias));
#else
typedef double Raw16Base __attribute__((ext_vector_type(2)));
typedef Raw16Base Raw16 __attribute__((may_alias));
#endif

inline Raw16 raw16_load(const void* p) { return *reinterpret_cast<const Raw16*>(p); }
inline void raw16_store(void* p, Raw16 v) { *reinterpret_cast<Raw16*>(p) = v; }

// The C tile out through local memory, for a group the 16-byte band store cannot serve (odd
// ldc, unaligned C, an edge). The stage is lo then hi as one column-major space of stride TileM,
// so a chunk of CW columns fits; per chunk `put(c0, cw, slot)` writes the thread's accumulators
// of tile columns [c0, c0 + cw) to *slot(column - c0, row), then the group stores the chunk with
// lane = consecutive row, so a warp's store is contiguous at any ldc. The caller's mainloop ends
// on a barrier after its last read of lo and hi; each chunk's barrier orders put before the
// read-back, and the one after it the read-back before the next chunk's put.
// evidence: docs/perf/gemm.md#gemm-wp-b-coalesced-c-epilogue-on-the-predicated-legs
template <int TileM, int TileN, int Threads, int LoElems, int HiElems, int Batch = 2, typename D, typename Put>
inline void staged_c_store(const sycl::nd_item<3>& item, D* lo, D* hi, D* Cb, int ldc, int m0, int n0, int m,
                           int n, int tid, D alpha, D beta, bool beta_zero, Put&& put) {
    static_assert(LoElems % TileM == 0 && HiElems % TileM == 0, "a stage column never straddles lo and hi");
    constexpr int CW = (LoElems + HiElems) / TileM;
    constexpr int Passes = (TileN + CW - 1) / CW;
    // Batch: prior loads in flight per thread. The accumulators are in the stage by now, so a
    // caller with registers to spare (wide v2) takes a whole chunk in one round trip, and only
    // that Batch switches to the row map below; any other Batch keeps the element map.
    // evidence: docs/perf/gemm.md#gemm-wp-q-a-one-round-staged-epilogue-for-wide-v2
    constexpr bool kWholeChunk = Batch > 2 && Threads % TileM == 0 && Batch * (Threads / TileM) == CW;
    auto slot = [&](int c, int r) -> D* {
        const int e = c * TileM + r;
        return e < LoElems ? lo + e : hi + (e - LoElems);
    };
#pragma unroll 1
    for (int pass = 0; pass < Passes; ++pass) {
        const int c0 = pass * CW;
        const int cw = TileN - c0 < CW ? TileN - c0 : CW;
        put(c0, cw, slot);
        item.barrier(sycl::access::fence_space::local_space);
        if constexpr (kWholeChunk) {
            // A thread keeps one row and every Step-th column; Batch columns per round trip. nc:
            // the chunk's columns inside C, none on a row outside it.
            constexpr int Step = Threads / TileM;
            const int r = tid % TileM;
            const int lim = n - n0 - c0 < cw ? n - n0 - c0 : cw;
            const int nc = m0 + r < m ? lim : 0;
            const std::ptrdiff_t step = static_cast<std::ptrdiff_t>(Step) * ldc;
#pragma unroll 1
            for (int cb = tid / TileM; cb < nc; cb += Batch * Step) {
                D* p = Cb + (m0 + r) + static_cast<std::ptrdiff_t>(n0 + c0 + cb) * ldc;
                D prior[Batch];
BATCHLAS_UNROLL_FULL
                for (int b = 0; b < Batch; ++b) prior[b] = D{};
                if (!beta_zero) {
BATCHLAS_UNROLL_FULL
                    for (int b = 0; b < Batch; ++b)
                        if (cb + b * Step < nc) prior[b] = p[b * step];
                }
BATCHLAS_UNROLL_FULL
                for (int b = 0; b < Batch; ++b)
                    if (cb + b * Step < nc)
                        p[b * step] = lin_epi(alpha, beta_zero ? D{} : beta, *slot(cb + b * Step, r), prior[b]);
            }
            if (pass + 1 < Passes) item.barrier(sycl::access::fence_space::local_space);
            continue;
        }
        const int count = TileM * cw;
        auto at = [&](int e, bool& ok) {
            const int row = m0 + e % TileM, col = n0 + c0 + e / TileM;
            ok = e < count && row < m && col < n;
            return row + static_cast<std::ptrdiff_t>(col) * ldc;
        };
#pragma unroll 1
        for (int e0 = tid; e0 < count; e0 += Batch * Threads) {
            D prior[Batch];
BATCHLAS_UNROLL_FULL
            for (int b = 0; b < Batch; ++b) {
                bool ok;
                const std::ptrdiff_t off = at(e0 + b * Threads, ok);
                prior[b] = (ok && !beta_zero) ? Cb[off] : D{};
            }
BATCHLAS_UNROLL_FULL
            for (int b = 0; b < Batch; ++b) {
                bool ok;
                const std::ptrdiff_t off = at(e0 + b * Threads, ok);
                if (ok) Cb[off] = lin_epi(alpha, beta_zero ? D{} : beta, *slot(0, e0 + b * Threads), prior[b]);
            }
        }
        if (pass + 1 < Passes) item.barrier(sycl::access::fence_space::local_space);
    }
}

struct Tile {
    static constexpr int M = 64;
    static constexpr int N = 64;
    static constexpr int K = 16;
    static constexpr int TT = 4;                           // thread tile, both directions
    static constexpr int LocalRows = M / TT;               // 16, the m direction
    static constexpr int LocalCols = N / TT;               // 16, the n direction
    static constexpr int Threads = LocalRows * LocalCols;  // 256
    // No padding: with an odd stride the compiler cannot prove 16-byte
    // alignment and every fragment load degrades to a scalar ld.shared.
    static constexpr int AStride = M;
    static constexpr int BStride = N;
    static constexpr int PerThreadA = M * K / Threads;  // 4
    static constexpr int PerThreadB = K * N / Threads;  // 4
};

static_assert(Tile::M * Tile::K == Tile::PerThreadA * Tile::Threads, "A staging");
static_assert(Tile::K * Tile::N == Tile::PerThreadB * Tile::Threads, "B staging");

}  // namespace wide_scalar

// OA/OB/K: wide v2's transpose form and slab depth (K16 in the name is historical); the NN
// kernel takes the defaults.
template <typename T, bool AlignedFastPath, int OA = 0, int OB = 0, int K = 16>
class GemmRegister64x64K16WideKernel;

// VecLen counts elements; the granule the fast path enforces is 16 bytes.
template <typename T>
inline bool can_use_64x64_k16_wide_fast_path(const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B,
                                             const MatrixView<T, MatrixFormat::Dense>& C) {
    constexpr int TileM = wide_scalar::Tile::M;
    constexpr int TileN = wide_scalar::Tile::N;
    constexpr int TileK = wide_scalar::Tile::K;
    constexpr int VecLen = 16 / static_cast<int>(sizeof(T));
    const auto m = A.rows();
    const auto k = A.cols();
    const auto n = B.cols();
    if ((m % TileM) != 0 || (n % TileN) != 0 || (k % TileK) != 0) {
        return false;
    }
    auto aligned = [](const T* p, int64_t ld, int64_t stride) {
        return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % 16u) == 0 &&
            (ld % VecLen) == 0 && (stride % VecLen) == 0;
    };
    return aligned(A.data_ptr(), A.ld(), A.stride()) &&
        aligned(B.data_ptr(), B.ld(), B.stride()) &&
        aligned(C.data_ptr(), C.ld(), C.stride());
}

// NN only: A is read as m x k and B as k x n directly, so a transposed operand
// has to fall back at the caller rather than be transposed here.
template <typename T, bool AlignedFastPath = false>
Event launch_register_64x64_k16_wide(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B,
                                     const MatrixView<T, MatrixFormat::Dense>& C,
                                     T alpha,
                                     T beta,
                                     const char* (*kernel_trace_name)(KernelVariant)) {
    BATCHLAS_KERNEL_TRACE_SCOPE(
        kernel_trace_name(KernelVariant::Tiled64x64RegisterK16Wide));

    using namespace wide_scalar;

    // std::complex is re-typed to the POD device type here, at the pointer
    // boundary, and never crosses into the kernel body -- alpha and beta too.
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int TileM = Tile::M;
    constexpr int TileN = Tile::N;
    constexpr int TileK = Tile::K;
    constexpr int TT = Tile::TT;
    constexpr int LocalRows = Tile::LocalRows;
    constexpr int LocalCols = Tile::LocalCols;
    constexpr int AStride = Tile::AStride;
    constexpr int BStride = Tile::BStride;

    constexpr int VecN = Vec16<D>::N;                 // elements per 16 bytes
    constexpr int Wb = (VecN < TT) ? VecN : TT;       // band width in elements
    constexpr int NB = TT / Wb;                       // number of bands
    constexpr int MSep = TileM / NB;                  // band separation, m
    constexpr int NSep = TileN / NB;                  // band separation, n
    constexpr int Chunks = Tile::PerThreadA / VecN;   // 16B chunks per stage
    constexpr int AGranPerRow = TileM / VecN;         // 16B granules per k-row of A

    static_assert(TileM % VecN == 0, "A tile must split into whole granules");
    static_assert(AGranPerRow * TileK == Chunks * Tile::Threads, "A granule handout");
    static_assert(TT % Wb == 0, "thread tile must split into whole bands");
    static_assert(Tile::PerThreadA % VecN == 0, "staging must be whole granules");
    static_assert(Tile::PerThreadA == Tile::PerThreadB, "Chunks is shared by A and B staging");
    static_assert(MSep % VecN == 0 && NSep % VecN == 0, "band base must stay aligned");
    static_assert(AStride % VecN == 0 && BStride % VecN == 0, "shared stride alignment");

    const int m = static_cast<int>(A.rows());
    const int k = static_cast<int>(A.cols());
    const int n = static_cast<int>(B.cols());

    const int gm_tiles = (m + TileM - 1) / TileM;
    const int gn_tiles = (n + TileN - 1) / TileN;

    // local_id(2) is the fastest-varying SYCL dimension; it carries m.
    const sycl::range<3> local(1, LocalCols, LocalRows);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(gn_tiles) * LocalCols,
                                static_cast<size_t>(gm_tiles) * LocalRows);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<Vec16<D>, 1> tile_a(sycl::range<1>(TileK * AStride / VecN), h);
        sycl::local_accessor<Vec16<D>, 1> tile_b(sycl::range<1>(TileK * BStride / VecN), h);

        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* b_ptr = reinterpret_cast<const D*>(B.data_ptr());
        D* c_ptr = reinterpret_cast<D*>(C.data_ptr());
        const int lda = static_cast<int>(A.ld());
        const int ldb = static_cast<int>(B.ld());
        const int ldc = static_cast<int>(C.ld());
        const int stride_a = static_cast<int>(A.stride());
        const int stride_b = static_cast<int>(B.stride());
        const int stride_c = static_cast<int>(C.stride());
        const int batch = static_cast<int>(A.batch_size());

        D alpha_d;
        D beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        batchlas::impl::parallel_for<GemmRegister64x64K16WideKernel<T, AlignedFastPath>>(h,
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch) {
                    return;
                }

                // m is the fastest-varying thread index: C is column-major, so
                // lanes differing in m touch consecutive addresses. Reversing it
                // makes the read of C one transaction per lane at beta != 0.
                const int ty = static_cast<int>(item.get_local_id(2));  // 0..15, m
                const int tx = static_cast<int>(item.get_local_id(1));  // 0..15, n
                const int tid = tx * LocalRows + ty;                    // linear local id

                const int m0 = static_cast<int>(item.get_group(2)) * TileM;
                const int n0 = static_cast<int>(item.get_group(1)) * TileN;

                const D* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                const D* Bb = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
                D* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;

                D* sa = reinterpret_cast<D*>(
                    tile_a.template get_multi_ptr<sycl::access::decorated::no>().get());
                D* sb = reinterpret_cast<D*>(
                    tile_b.template get_multi_ptr<sycl::access::decorated::no>().get());

                // Plain local array, fully unrolled: this is the register file.
                D accum[TT][TT];
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < TT; ++i) {
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < TT; ++j) {
                        accum[i][j] = D{};
                    }
                }

                // A is column-major m x k. Granule ids go out in linear tid
                // order so consecutive lanes take consecutive 16-byte granules.
                // Handing each thread PerThreadA consecutive *elements*, as the
                // float kernels do, is a 4- to 8-way shared-store conflict here.
                int a_gm[Chunks];
                int a_gk[Chunks];
BATCHLAS_UNROLL_FULL
                for (int c = 0; c < Chunks; ++c) {
                    const int g = tid + c * Tile::Threads;
                    a_gm[c] = (g % AGranPerRow) * VecN;
                    a_gk[c] = g / AGranPerRow;
                }

                // B is scattered into shared as [k][n]. The 4-way bank conflict
                // on the store is deliberate: it keeps the read of B coalesced.
                const int b_k = (tid % (TileK / Tile::PerThreadB)) * Tile::PerThreadB;
                const int b_n = tid / (TileK / Tile::PerThreadB);

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    if constexpr (AlignedFastPath) {
BATCHLAS_UNROLL_FULL
                        for (int c = 0; c < Chunks; ++c) {
                            raw16_store(
                                &sa[a_gk[c] * AStride + a_gm[c]],
                                raw16_load(Ab + (m0 + a_gm[c]) +
                                           static_cast<std::ptrdiff_t>(k0 + a_gk[c]) * lda));
                        }
                        alignas(16) D vb[Tile::PerThreadB];
BATCHLAS_UNROLL_FULL
                        for (int c = 0; c < Chunks; ++c) {
                            raw16_store(&vb[c * VecN],
                                        raw16_load(Bb + (k0 + b_k + c * VecN) +
                                                   static_cast<std::ptrdiff_t>(n0 + b_n) * ldb));
                        }
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < Tile::PerThreadB; ++i) {
                            sb[(b_k + i) * BStride + b_n] = vb[i];
                        }
                    } else {
                        // Predicated staging fills the tile to its full 64x16,
                        // zero outside the matrix, so the inner loop below needs
                        // no bounds checks and is identical in both paths.
BATCHLAS_UNROLL_FULL
                        for (int c = 0; c < Chunks; ++c) {
                            const int gk_a = k0 + a_gk[c];
BATCHLAS_UNROLL_FULL
                            for (int e = 0; e < VecN; ++e) {
                                const int gm = m0 + a_gm[c] + e;
                                sa[a_gk[c] * AStride + a_gm[c] + e] =
                                    (gm < m && gk_a < k)
                                        ? Ab[gm + static_cast<std::ptrdiff_t>(gk_a) * lda]
                                        : D{};
                            }
                        }
                        const int gn_b = n0 + b_n;
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < Tile::PerThreadB; ++i) {
                            const int gk = k0 + b_k + i;
                            sb[(b_k + i) * BStride + b_n] =
                                (gk < k && gn_b < n)
                                    ? Bb[gk + static_cast<std::ptrdiff_t>(gn_b) * ldb]
                                    : D{};
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

BATCHLAS_UNROLL_FULL
                    for (int kk = 0; kk < TileK; ++kk) {
                        D af[TT];
                        D bf[TT];
BATCHLAS_UNROLL_FULL
                        for (int band = 0; band < NB; ++band) {
                            const Vec16<D> t =
                                vec_ref(&sa[kk * AStride + band * MSep + ty * Wb]);
BATCHLAS_UNROLL_FULL
                            for (int w = 0; w < Wb; ++w) {
                                af[band * Wb + w] = t.v[w];
                            }
                        }
BATCHLAS_UNROLL_FULL
                        for (int band = 0; band < NB; ++band) {
                            const Vec16<D> t =
                                vec_ref(&sb[kk * BStride + band * NSep + tx * Wb]);
BATCHLAS_UNROLL_FULL
                            for (int w = 0; w < Wb; ++w) {
                                bf[band * Wb + w] = t.v[w];
                            }
                        }
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < TT; ++i) {
BATCHLAS_UNROLL_FULL
                            for (int j = 0; j < TT; ++j) {
                                fma_acc(accum[i][j], af[i], bf[j]);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Within a band the Wb rows are consecutive in m, so a whole
                // band is one 16-byte access into column-major C. That needs only
                // this tile inside C and 16-byte columns, per group, whatever A and
                // B are; any other group of the predicated leg is staged (lanes Wb
                // rows apart store one element each), except at VecN == 1, where
                // that scalar store is already the band store.
                const bool beta_zero = dev_is_zero(beta_d);
                const bool c_vec = AlignedFastPath ||
                    (m0 + TileM <= m && n0 + TileN <= n && ldc % VecN == 0 &&
                     reinterpret_cast<std::uintptr_t>(Cb) % 16u == 0);
                constexpr bool kVecOnly = AlignedFastPath || VecN > 1;  // past the staged return
                if constexpr (!AlignedFastPath && VecN > 1) {
                    if (!c_vec) {
                        staged_c_store<TileM, TileN, Tile::Threads, TileK * AStride, TileK * BStride>(
                            item, sa, sb, Cb, ldc, m0, n0, m, n, tid, alpha_d, beta_d, beta_zero,
                            [&](int c0, int cw, auto slot) {
BATCHLAS_UNROLL_FULL
                                for (int bm = 0; bm < NB; ++bm) {
BATCHLAS_UNROLL_FULL
                                    for (int j = 0; j < TT; ++j) {
                                        const int c = (j / Wb) * NSep + tx * Wb + j % Wb - c0;
                                        if (c < 0 || c >= cw) continue;
                                        Vec16<D> v;
BATCHLAS_UNROLL_FULL
                                        for (int w = 0; w < Wb; ++w) v.v[w] = accum[bm * Wb + w][j];
                                        vec_ref(slot(c, bm * MSep + ty * Wb)) = v;
                                    }
                                }
                            });
                        return;
                    }
                }
BATCHLAS_UNROLL_FULL
                for (int bm = 0; bm < NB; ++bm) {
                    const int gm = m0 + bm * MSep + ty * Wb;
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < TT; ++j) {
                        const int bn = j / Wb;
                        const int w_n = j % Wb;
                        const int gn = n0 + bn * NSep + tx * Wb + w_n;
                        if (kVecOnly || c_vec) {
                            D* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                            Vec16<D> out;
                            if (beta_zero) {
BATCHLAS_UNROLL_FULL
                                for (int w = 0; w < Wb; ++w) {
                                    out.v[w] =
                                        lin_epi(alpha_d, D{}, accum[bm * Wb + w][j], D{});
                                }
                            } else {
                                const Vec16<D> prior = vec_ref(const_cast<const D*>(p));
BATCHLAS_UNROLL_FULL
                                for (int w = 0; w < Wb; ++w) {
                                    out.v[w] = lin_epi(alpha_d, beta_d,
                                                       accum[bm * Wb + w][j], prior.v[w]);
                                }
                            }
                            vec_ref(p) = out;
                        } else {
                            if (gn >= n) {
                                continue;
                            }
BATCHLAS_UNROLL_FULL
                            for (int w = 0; w < Wb; ++w) {
                                const int row = gm + w;
                                if (row >= m) {
                                    continue;
                                }
                                D* p = &Cb[row + static_cast<std::ptrdiff_t>(gn) * ldc];
                                const D prior = beta_zero ? D{} : *p;
                                *p = lin_epi(alpha_d, beta_zero ? D{} : beta_d,
                                             accum[bm * Wb + w][j], prior);
                            }
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

// ---- wide v2: every transpose form, float, complex<float> and complex<double> ------------
//
// One 64x64 kernel per (op(A), op(B)) over {N, T, C} and slab depth 16 or 8. Staging is a pure
// copy into one shared layout, s[kk][i] in 16-byte granules (i = m for A, n for B); conjugation
// is folded into the FMA signs. Two buffers, one barrier per slab, the next slab prefetched into
// ext_vector registers. The leg (16-byte or element global loads) is per launch; whether a
// slab is predicated is per work-group: only an edge group or the k tail pays for bounds.
// evidence: docs/perf/gemm.md#gemm-wide-v2-one-double-buffered-64x64-kernel-for-every-form

// .reqntid: without it ptxas ignores LAUNCH_BOUNDS' .minnctapersm on this functor (that
// max_work_group_size emits no .maxntid here). The launch is (1, 1, 256) exactly.
// evidence: docs/perf/gemm.md#gemm-wp-q-the-launch-bound-was-ignored
#if !BATCHLAS_SYCL_IMPL_ACPP && defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#define BATCHLAS_WIDE_V2_REQD_WG(min_groups) \
    [[sycl::reqd_work_group_size(1, 1, 256), intel::min_work_groups_per_cu(min_groups)]]
#else
#define BATCHLAS_WIDE_V2_REQD_WG(min_groups)
#endif

template <typename T>
inline constexpr bool kWide64V2 = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>> ||
                                  std::is_same_v<T, std::complex<double>>;

namespace wide_v2 {

using batchlas::sycl_device::Cx;

// The XOR applied to the granule index of row kk, identically on store and fragment read; CF
// floats per element. It makes the k-contiguous operand's transposing store conflict-free: one
// bank-conflict set (an 8-lane phase of a 16-byte store, a warp of float ones) spans 4 row groups
// (kk / VecN) x 2 adjacent granules (GemmWide64Body::first_k), so bits 1-2 follow the row
// group. For CF <= 2 that is four values, so the fragment read needs four offsets. Float at TileK = 8 has two
// row groups x 4 granules per warp, so its row group flips bit 2. complex<double> (one element
// per granule) puts 8 consecutive kk in a phase, so kk & 7 spreads them over all 32 banks.
// evidence: docs/perf/gemm.md#gemm-wp-j-complex-double-on-wide-v2
template <int CF, int TileK>
constexpr int swz(int kk) {
    if constexpr (CF == 4) return kk & 7;
    else if constexpr (CF == 1 && TileK == 8) return ((kk / 4) & 1) << 2;
    else return ((kk / (4 / CF)) & 3) << 1;
}

typedef float EF2 __attribute__((ext_vector_type(2), may_alias));

inline Vec4<float> ld16(const float* p) {
    return *reinterpret_cast<const Vec4<float>*>(__builtin_assume_aligned(p, 16));
}
inline void st16(float* p, Vec4<float> v) {
    *reinterpret_cast<Vec4<float>*>(__builtin_assume_aligned(p, 16)) = v;
}

// Real: one FMA. Complex: a' = conj(a) when CA, b' = conj(b) when CB, as four FMAs with the
// signs folded (bit-identical to conjugating at staging): re += ar br - sA sB ai bi,
// im += sB ar bi + sA ai br.
template <bool CA, bool CB>
inline void fma_signed(float& acc, float a, float b) {
    acc = sycl::fma(a, b, acc);
}
template <bool CA, bool CB, typename R>
inline void fma_signed(Cx<R>& acc, Cx<R> a, Cx<R> b) {
    acc.re = sycl::fma(a.re, b.re, acc.re);
    acc.re = sycl::fma(CA != CB ? a.im : -a.im, b.im, acc.re);
    acc.im = sycl::fma(CB ? -a.re : a.re, b.im, acc.im);
    acc.im = sycl::fma(CA ? -a.im : a.im, b.re, acc.im);
}

// complex<double> is the whole granule, bit-cast (a register move; acpp leaves a memcpy'd private
// copy of a 16-byte complex<double> on the stack). Metal's struct Vec4 is not trivially copyable.
template <typename D>
inline D elem(const Vec4<float>& t, int w) {
    if constexpr (std::is_same_v<D, float>) {
        return t[w];
    } else if constexpr (std::is_same_v<D, Cx<float>>) {
        return D{t[2 * w], t[2 * w + 1]};
    } else if constexpr (std::is_trivially_copyable_v<Vec4<float>>) {
        return __builtin_bit_cast(D, t);
    } else {
        const float f[4] = {t[0], t[1], t[2], t[3]};
        D d;
        __builtin_memcpy(&d, f, sizeof(D));
        return d;
    }
}

// One operand's slab into CF 16-byte registers. A thread's elements, relative to its first
// one p: KC (contiguous along k: A^T, B) slot q of register c is (kk = kb + q, i = ib + c) at
// c ld + q; otherwise (along i) it is (kk = kb + c KStep, i = ib + q) at c KStep ld + q.
// Pred zero-fills outside k and i_lim, so the inner loop needs no bounds. KStep: the rows one
// pass of an operand's staging lanes covers along i.
template <bool Pred, bool VecLoads, bool KC, int CF, int KStep>
inline void gload(Vec4<float> (&r)[CF], const float* p, int ld, int k0, int kb, int ib, int k, int i_lim) {
    constexpr int VecN = 4 / CF;
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < CF; ++c) {
        const std::ptrdiff_t base = KC ? static_cast<std::ptrdiff_t>(c) * ld + k0
                                       : (static_cast<std::ptrdiff_t>(c) * KStep + k0) * ld;
        if constexpr (VecLoads && !Pred) {
            r[c] = ld16(p + base * CF);
        } else {
BATCHLAS_UNROLL_FULL
            for (int q = 0; q < VecN; ++q) {
                const bool ok = !Pred || (k0 + kb + (KC ? q : c * KStep) < k && ib + (KC ? c : q) < i_lim);
                const float* e0 = p + (base + q) * CF;
                // complex<float>: one 8-byte load per element where the lane's base allows it.
#if !BATCHLAS_DEVICE_NO_VECTOR_TYPES
                if (CF == 2 && (reinterpret_cast<std::uintptr_t>(p) & 7u) == 0) {
                    const EF2 t = ok ? *reinterpret_cast<const EF2*>(__builtin_assume_aligned(e0, 8)) : EF2{0.0f, 0.0f};
                    r[c][q * CF] = t[0];
                    r[c][q * CF + CF - 1] = t[1];
                    continue;
                }
#endif
BATCHLAS_UNROLL_FULL
                for (int e = 0; e < CF; ++e) r[c][q * CF + e] = ok ? e0[e] : 0.0f;
            }
        }
    }
}

// The registers into shared row kk (RowE elements wide) at granule (i / VecN) ^ swz(kk). Along i, a
// register is one granule. Along k, complex<float> pairs its two columns into one granule per row;
// float stores four scalars, one per row; complex<double> stores its CF columns of row kb.
template <bool KC, int CF, int KStep, int TileK, int RowE>
inline void sstore(float* s, const Vec4<float> (&r)[CF], int kb, int ib) {
    constexpr int VecN = 4 / CF, RowF = RowE * CF;
    if constexpr (!KC) {
BATCHLAS_UNROLL_FULL
        for (int c = 0; c < CF; ++c) {
            const int kk = kb + c * KStep;
            st16(s + kk * RowF + (((ib / VecN) ^ swz<CF, TileK>(kk)) << 2), r[c]);
        }
    } else if constexpr (CF == 4) {
BATCHLAS_UNROLL_FULL
        for (int c = 0; c < CF; ++c) st16(s + kb * RowF + (((ib + c) ^ swz<CF, TileK>(kb)) << 2), r[c]);
    } else if constexpr (CF == 2) {
BATCHLAS_UNROLL_FULL
        for (int q = 0; q < 2; ++q) {
            const int kk = kb + q;
            const Vec4<float> v{r[0][2 * q], r[0][2 * q + 1], r[1][2 * q], r[1][2 * q + 1]};
            st16(s + kk * RowF + (((ib >> 1) ^ swz<CF, TileK>(kk)) << 2), v);
        }
    } else {
BATCHLAS_UNROLL_FULL
        for (int q = 0; q < 4; ++q) {
            const int kk = kb + q;
            s[kk * RowF + (((ib >> 2) ^ swz<CF, TileK>(kk)) << 2) + (ib & 3)] = r[0][q];
        }
    }
}

// The element leg's k-contiguous operand (float, complex<float>): lane st takes row kb = st % TileK
// of columns ib + edi(j), so TileK lanes walk one column run and a load instruction is a few whole
// sectors (the 16-byte-granule map put 4 kk per lane and sent ~4x the sectors). Its shared row is
// padded by 4 floats and unswizzled; at TileK = 16 rows 8-15 also flip the element's slot in its
// granule (ekx), so the TileK x 2 transposing store hits 32 banks. complex<float> moves 8 bytes per
// element where the lane's base is 8-byte aligned.
// evidence: docs/perf/gemm.md#gemm-wp-y-coalesced-element-leg-lane-maps
template <int OpThreads, int TileK>
constexpr int edi(int j) {
    return (OpThreads / TileK) * j;
}
template <int CF, int TileK>
constexpr int ekx(int kk) {
    return TileK == 16 && (kk & 8) ? 2 / CF : 0;
}

template <bool Pred, int CF, int OpThreads, int TileK>
inline void eload(Vec4<float> (&r)[CF], const float* p, int ld, int k0, int kb, int ib, int k, int i_lim) {
    const bool al8 = CF == 2 && (reinterpret_cast<std::uintptr_t>(p) & 7u) == 0;
    // The column step is rebuilt per call: hoisted, its multiples spilled at the cap. 64-bit:
    // 16 ld CF passes INT_MAX at ld ~ 67M (complex<float>).
    std::ptrdiff_t step = static_cast<std::ptrdiff_t>(edi<OpThreads, TileK>(1)) * ld * CF;
#if !BATCHLAS_SYCL_IMPL_ACPP && defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("" : "+l"(step));
#endif
    const float* q = p + k0 * CF;
BATCHLAS_UNROLL_FULL
    for (int j = 0; j < 4; ++j, q += step) {
        const bool ok = !Pred || (k0 + kb < k && ib + edi<OpThreads, TileK>(j) < i_lim);
        float v[CF];
#if !BATCHLAS_DEVICE_NO_VECTOR_TYPES
        if (CF == 2 && al8) {
            const EF2 t = ok ? *reinterpret_cast<const EF2*>(__builtin_assume_aligned(q, 8)) : EF2{0.0f, 0.0f};
            v[0] = t[0];
            v[CF - 1] = t[1];
        } else
#endif
        {
BATCHLAS_UNROLL_FULL
            for (int e = 0; e < CF; ++e) v[e] = ok ? q[e] : 0.0f;
        }
BATCHLAS_UNROLL_FULL
        for (int e = 0; e < CF; ++e) r[j * CF / 4][(j * CF) % 4 + e] = v[e];
    }
}

template <int CF, int OpThreads, int TileK, int RowF>
inline void estore(float* s, const Vec4<float> (&r)[CF], int kb, int ib) {
    constexpr int VecN = 4 / CF;
BATCHLAS_UNROLL_FULL
    for (int j = 0; j < 4; ++j) {
        const int i = ib + edi<OpThreads, TileK>(j);
        float* d = s + kb * RowF + ((i / VecN) << 2) + ((i % VecN) ^ ekx<CF, TileK>(kb)) * CF;
        const int q = (j * CF) % 4;
#if BATCHLAS_DEVICE_NO_VECTOR_TYPES
BATCHLAS_UNROLL_FULL
        for (int e = 0; e < CF; ++e) d[e] = r[j * CF / 4][q + e];
#else
        if constexpr (CF == 1) *d = r[0][j];
        else *reinterpret_cast<EF2*>(__builtin_assume_aligned(d, 8)) = EF2{r[j / 2][q], r[j / 2][q + 1]};
#endif
    }
}

// One operand's slab of a complex<float> lane (CF = 2) straight into the sstore layout; p is the
// lane's first element of this slab. Along i a granule is two consecutive elements: one 16-byte
// copy on the 16-byte leg, src-size cut to the elements inside i_lim, else two 8-byte ones. Along
// k the granule pairs two columns, so every element is its own 8-byte copy: the bases must be
// 8-byte aligned (the launcher checks; alignof(std::complex<float>) is 4).
// evidence: docs/perf/gemm.md#gemm-wp-v-three-copy-async-stages-for-the-128x128-complex-float-tile
template <bool VecLoads, bool KC, int KStep, int TileK, int RowE>
inline void acopy(float* s, std::uint32_t s_l, const float* p, int ld, int k0, int kb, int ib, int ibs, int k,
                  int i_lim) {
    constexpr int CF = 2, RowF = RowE * CF;
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < CF; ++c) {
        if constexpr (!KC) {
            const int kk = kb + c * KStep;
            const int o = kk * RowF + (((ibs >> 1) ^ swz<CF, TileK>(kk)) << 2);
            const float* src = p + static_cast<std::ptrdiff_t>(c) * KStep * ld * CF;
            const bool kin = k0 + kk < k;
            if constexpr (VecLoads) {
                const int left = i_lim - ib;
                copy_async_zfill<16>(s + o, s_l + 4u * o, src, !kin || left <= 0 ? 0u : left == 1 ? 8u : 16u);
            } else {
BATCHLAS_UNROLL_FULL
                for (int q = 0; q < 2; ++q)
                    copy_async_zfill<8>(s + o + 2 * q, s_l + 4u * (o + 2 * q), src + q * CF,
                                        kin && ib + q < i_lim ? 8u : 0u);
            }
        } else {
BATCHLAS_UNROLL_FULL
            for (int q = 0; q < 2; ++q) {
                const int kk = kb + q;
                const int o = kk * RowF + (((ibs >> 1) ^ swz<CF, TileK>(kk)) << 2) + 2 * c;
                copy_async_zfill<8>(s + o, s_l + 4u * o, p + (static_cast<std::ptrdiff_t>(c) * ld + q) * CF,
                            k0 + kk < k && ib + c < i_lim ? 8u : 0u);
            }
        }
    }
}

}  // namespace wide_v2

// A FUNCTOR, not a lambda, only so the launch bound can be spelled. Two groups per CU for
// complex<float> (its prefetch registers do not fit three groups' 85-register cap), one for
// complex<double> (64 accumulator registers; FP64 is compute-bound at 8 warps per SM).
// TileK = 8 halves the slab; each lane still stages CF granules, so warps 0-3 stage A and 4-7 B.
// 128x128 (complex<float> only, WP-O; one group per CU): each operand is two 64-wide halves,
// staged with the 64x64 lane map, KStep and swizzle, the second 64 down i; the thread tile is 8x8.
// evidence: docs/perf/gemm.md#gemm-wp-o-a-128x128-complex-float-tile-on-wide-v2
template <typename T, Transpose OpA, Transpose OpB, bool VecLoads, int TileK_, int TileM_ = 64, int TileN_ = 64,
          int Stages_ = 2>
struct GemmWide64Body {
    using D = typename wide_scalar::DevMap<T>::type;
    static constexpr int TileM = TileM_, TileN = TileN_, TileK = TileK_, Threads = 256;
    // Stages > 2: that many local slabs filled by cp.async, Stages - 1 in flight, no register prefetch.
    static constexpr int Stages = Stages_;
    static constexpr bool kPipe = Stages > 2;
    static constexpr int TTM = TileM / 16, TTN = TileN / 16;    // thread tile over 16 x 16 lanes
    static constexpr int CF = static_cast<int>(sizeof(D) / sizeof(float));
    static constexpr int VecN = 4 / CF;
    static constexpr int NBM = TTM / VecN, NBN = TTN / VecN;     // fragment granules per operand
    static constexpr int MSep = TileM / NBM, NSep = TileN / NBN; // band separation
    static constexpr int HA = TileM / 64, HB = TileN / 64;       // 64-wide staging halves
    static constexpr bool KA = OpA != Transpose::NoTrans, KB = OpB == Transpose::NoTrans;
    // The element leg's coalesced lane maps (wide_v2::eload); PA/PB: a padded, unswizzled KC row.
    static constexpr bool kElem = !VecLoads && CF <= 2 && !kPipe;
    static constexpr bool PA = kElem && KA, PB = kElem && KB;
    template <bool KC>
    static constexpr bool kKF = kElem && KC;  // wide_v2::eload's k-fastest map
    static constexpr int RowA = TileM * CF + (PA ? 4 : 0), RowB = TileN * CF + (PB ? 4 : 0);  // floats
    static constexpr int BufA = TileK * RowA, BufB = TileK * RowB;
    static constexpr int GPR = 64 / VecN;                        // granules per row of one half
    static constexpr bool Split = TileK == 8;
    static constexpr int OpThreads = Split ? Threads / 2 : Threads;  // staging lanes per operand
    static constexpr int KStep = OpThreads / GPR;
    // Prefetch sets: A's halves then B's; under Split a lane holds only its own operand's.
    static constexpr int NR = Split ? (HA > HB ? HA : HB) : HA + HB, RB0 = Split ? 0 : HA;
    static constexpr bool CA = batchlas::sycl_device::dev_is_complex_v<D> && OpA == Transpose::ConjTrans;
    static constexpr bool CB = batchlas::sycl_device::dev_is_complex_v<D> && OpB == Transpose::ConjTrans;
    static constexpr int kMinGroupsPerCu = (CF == 4 || TileM * TileN > 4096) ? 1 : (CF == 1 ? 3 : 2);
    // Prior loads per thread in flight in the staged epilogue: its whole chunk.
    static constexpr int EpiBatch = 2 * TileK * (TileM + TileN) / Threads;
    static_assert(kWide64V2<T> && ((TileM == 64 && TileN == 64) || (CF == 2 && TileM % 64 == 0 && TileN % 64 == 0)),
                  "64x64 for every v2 scalar; larger tiles are whole 64-wide halves, complex<float> only");
    static_assert(TTM % VecN == 0 && TTN % VecN == 0 && EpiBatch * Threads == 2 * TileK * (TileM + TileN),
                  "whole fragment granules and a whole-chunk epilogue batch");
    static_assert(TileK == 16 || TileK == 8, "first_k and swz map these two slab depths");
    static_assert(TileM * TileN == 4096 || TileK == 8, "a larger tile at k = 16 needs > 48 KiB and NR = 4");
    static_assert(GPR * TileK == CF * OpThreads, "CF granules per staging lane per operand");
    static_assert(!kPipe || (CF == 2 && Split && HA == 2 && HB == 2),
                  "the cp.async slabs are complex<float> 128x128 at TileK = 8 (wide_v2::acopy, one half per lane)");

    const float* a_ptr;
    const float* b_ptr;
    D* c_ptr;
    int m, n, k, lda, ldb, ldc;
    std::ptrdiff_t stride_a, stride_b, stride_c;
    int batch;
    D alpha, beta;
    sycl::local_accessor<wide_scalar::Vec16<float>, 1> tile_a, tile_b;

    // A thread's first staged element (kb, ib) of an operand. Along i: granule tid % GPR of row
    // tid / GPR (the next one KStep rows down). Along k: CF columns of VecN k, row group kq;
    // complex<float> takes kq from tid bits 0, 1, 3 and its column pair from the rest, so an
    // 8-lane phase holds 4 row groups x 2 column pairs (the swizzle's premise, wide_v2::swz).
    // At TileK = 8 there are only 4 row groups (complex<float>) or 2 (float): tid bits 0-1 / 0.
    // complex<double> (VecN = 1): kk = tid % TileK, so a phase is 8 consecutive kk of one column.
    // tid is the lane's index among its operand's OpThreads staging lanes.
    static constexpr int KG = TileK / VecN;  // row groups along k
    template <bool KC>
    static int first_k(int tid) {
        if constexpr (kKF<KC>) return tid % TileK;
        else if constexpr (!KC) return tid / GPR;
        else if constexpr (CF != 2 || TileK == 8) return (tid % KG) * VecN;
        else return ((tid & 3) | ((tid >> 1) & 4)) * VecN;
    }
    template <bool KC>
    static int first_i(int tid) {
        if constexpr (kKF<KC>) return tid / TileK;
        else if constexpr (!KC) return (tid % GPR) * VecN;
        else if constexpr (CF != 2 || TileK == 8) return (tid / KG) * CF;
        else return (((tid >> 2) & 1) | ((tid >> 3) & ~1)) * CF;
    }

    template <bool KC>
    __attribute__((always_inline)) static void fetch(Vec4<float> (&r)[CF], const float* p, int ld, int k0, int kb,
                                                     int ib, int k, int lim, bool full) {
        if constexpr (kKF<KC>) {
            if (full) wide_v2::eload<false, CF, OpThreads, TileK>(r, p, ld, k0, kb, ib, k, lim);
            else wide_v2::eload<true, CF, OpThreads, TileK>(r, p, ld, k0, kb, ib, k, lim);
        } else if (full) {
            wide_v2::gload<false, VecLoads, KC, CF, KStep>(r, p, ld, k0, kb, ib, k, lim);
        } else {
            wide_v2::gload<true, VecLoads, KC, CF, KStep>(r, p, ld, k0, kb, ib, k, lim);
        }
    }
    template <bool KC, int RowF, int RowE>
    __attribute__((always_inline)) static void put(float* s, const Vec4<float> (&r)[CF], int kb, int ib) {
        if constexpr (kKF<KC>) wide_v2::estore<CF, OpThreads, TileK, RowF>(s, r, kb, ib);
        else wide_v2::sstore<KC, CF, KStep, TileK, RowE>(s, r, kb, ib);
    }
    // The fragment read's granule XOR: none on a padded row.
    template <bool P>
    static constexpr int sw(int kk) {
        return P ? 0 : wide_v2::swz<CF, TileK>(kk);
    }

    // always_inline: left out of line (complex<float>), the body spilled behind a call.
    BATCHLAS_WIDE_V2_REQD_WG(kMinGroupsPerCu)
    __attribute__((always_inline)) void operator()(sycl::nd_item<3> item) const {
        using namespace wide_v2;
        const int bid = static_cast<int>(item.get_group(0));
        if (bid >= batch) return;
        const int tid = static_cast<int>(item.get_local_id(2));
        const int ty = tid & 15, tx = tid >> 4;  // m, n: lanes one apart are one granule apart in m
        const int m0 = static_cast<int>(item.get_group(1)) * TileM;
        const int n0 = static_cast<int>(item.get_group(2)) * TileN;
        D* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
        float* sa = reinterpret_cast<float*>(tile_a.template get_multi_ptr<sycl::access::decorated::no>().get());
        float* sb = reinterpret_cast<float*>(tile_b.template get_multi_ptr<sycl::access::decorated::no>().get());

        const int st = tid % OpThreads;
        const bool stage_a = !Split || tid < OpThreads;  // warp-uniform
        const int a_kb = first_k<KA>(st), a_ib = first_i<KA>(st);
        const int b_kb = first_k<KB>(st), b_ib = first_i<KB>(st);
        auto origin = [](int i, int kk, int ld, bool kc) {
            return kc ? static_cast<std::ptrdiff_t>(i) * ld + kk : i + static_cast<std::ptrdiff_t>(kk) * ld;
        };
        const float* pa = a_ptr + (static_cast<std::ptrdiff_t>(bid) * stride_a + origin(m0 + a_ib, a_kb, lda, KA)) * CF;
        const float* pb = b_ptr + (static_cast<std::ptrdiff_t>(bid) * stride_b + origin(n0 + b_ib, b_kb, ldb, KB)) * CF;
        const bool a_in = m0 + TileM <= m, b_in = n0 + TileN <= n;
        const std::ptrdiff_t a_half = (KA ? static_cast<std::ptrdiff_t>(64) * lda : 64) * CF;
        const std::ptrdiff_t b_half = (KB ? static_cast<std::ptrdiff_t>(64) * ldb : 64) * CF;

        D accum[TTM][TTN];
BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TTM; ++i) {
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TTN; ++j) accum[i][j] = D{};
        }

        Vec4<float> r[NR][CF];
        // The halves unroll by static_for, not a 1-trip loop: LICM on that loop hoisted the edge
        // tests and spilled float 64x64x16 NC on sm_89.
        auto load = [&](int k0) __attribute__((always_inline)) {
            const bool kin = k0 + TileK <= k;
            if (stage_a)
                static_for<HA>([&](auto h) {
                    fetch<KA>(r[h], pa + h * a_half, lda, k0, a_kb, a_ib, k, m - m0 - 64 * h, a_in && kin);
                });
            if (!Split || !stage_a)
                static_for<HB>([&](auto h) {
                    fetch<KB>(r[RB0 + h], pb + h * b_half, ldb, k0, b_kb, b_ib, k, n - n0 - 64 * h, b_in && kin);
                });
        };
        auto store = [&](int buf) __attribute__((always_inline)) {
            if (stage_a)
                static_for<HA>([&](auto h) { put<KA, RowA, TileM>(sa + buf * BufA, r[h], a_kb, a_ib + 64 * h); });
            if (!Split || !stage_a)
                static_for<HB>([&](auto h) { put<KB, RowB, TileN>(sb + buf * BufB, r[RB0 + h], b_kb, b_ib + 64 * h); });
        };
        auto compute = [&](int buf) __attribute__((always_inline)) {
            const float* xa = sa + buf * BufA;
            const float* xb = sb + buf * BufB;
BATCHLAS_UNROLL_FULL
            for (int kk = 0; kk < TileK; ++kk) {
                D af[TTM], bf[TTN];
BATCHLAS_UNROLL_FULL
                for (int band = 0; band < NBM; ++band) {
                    const Vec4<float> ta =
                        ld16(xa + kk * RowA + (((band * (MSep / VecN) + ty) ^ sw<PA>(kk)) << 2));
BATCHLAS_UNROLL_FULL
                    for (int w = 0; w < VecN; ++w) af[band * VecN + w] = elem<D>(ta, w ^ (PA ? ekx<CF, TileK>(kk) : 0));
                }
BATCHLAS_UNROLL_FULL
                for (int band = 0; band < NBN; ++band) {
                    const Vec4<float> tb =
                        ld16(xb + kk * RowB + (((band * (NSep / VecN) + tx) ^ sw<PB>(kk)) << 2));
BATCHLAS_UNROLL_FULL
                    for (int w = 0; w < VecN; ++w) bf[band * VecN + w] = elem<D>(tb, w ^ (PB ? ekx<CF, TileK>(kk) : 0));
                }
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < TTM; ++i) {
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < TTN; ++j) fma_signed<CA, CB>(accum[i][j], af[i], bf[j]);
                }
            }
        };

        // One barrier per slab, as in the 128x128 kernel: slab s+1 is stored into the buffer
        // slab s-1 was read from, and every read of it finished before the barrier ending s-1.
        const int slabs = (k + TileK - 1) / TileK;
        if constexpr (kPipe) {
            // As the 128x128 u4 pipeline: step s waits for slab s (own copies, then the barrier for
            // everyone's), refills the buffer slab s - 1 was read from and computes. One commit per
            // step, empty past the end, keeps the wait count at Stages - 2. Each operand pointer
            // bumps one slab per issue; the buffer index is runtime (an unrolled ring is Stages x the code).
            std::uint32_t sa_l = 0, sb_l = 0;
#if BATCHLAS_GEMM_CP_ASYNC
            asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(sa_l) : "l"(sa));
            asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(sb_l) : "l"(sb));
#endif
            // Every lane copies one 64-wide half of both operands (lanes 0-127 the first), so all
            // eight warps share the copies; split by operand, four warps issued them all (slower).
            const int hh = tid / OpThreads;  // warp-uniform
            const float* ga = pa + hh * a_half;
            const float* gb = pb + hh * b_half;
            const int a_lim = m - m0 - 64 * hh, b_lim = n - n0 - 64 * hh;
            auto issue = [&](int k0, int buf) __attribute__((always_inline)) {
                wide_v2::acopy<VecLoads, KA, KStep, TileK, TileM>(sa + buf * BufA, sa_l + 4u * (buf * BufA), ga, lda,
                                                                 k0, a_kb, a_ib, a_ib + 64 * hh, k, a_lim);
                wide_v2::acopy<VecLoads, KB, KStep, TileK, TileN>(sb + buf * BufB, sb_l + 4u * (buf * BufB), gb, ldb,
                                                                 k0, b_kb, b_ib, b_ib + 64 * hh, k, b_lim);
                ga += (KA ? TileK : static_cast<std::ptrdiff_t>(TileK) * lda) * CF;
                gb += (KB ? TileK : static_cast<std::ptrdiff_t>(TileK) * ldb) * CF;
            };
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < Stages - 1; ++j) {
                if (j < slabs) issue(j * TileK, j);
                copy_async_commit();
            }
            int buf = 0;
#pragma unroll 1
            for (int s = 0; s < slabs; ++s) {
                copy_async_wait<Stages - 2>();
                item.barrier(sycl::access::fence_space::local_space);
                const int nb = buf == 0 ? Stages - 1 : buf - 1;
                if (s + Stages - 1 < slabs) issue((s + Stages - 1) * TileK, nb);
                copy_async_commit();
                compute(buf);
                buf = buf + 1 == Stages ? 0 : buf + 1;
            }
            copy_async_wait<0>();
        } else {
            load(0);
            store(0);
            item.barrier(sycl::access::fence_space::local_space);
            int s = 0;
            for (; s + 1 < slabs; s += 2) {
                load((s + 1) * TileK);
                compute(0);
                store(1);
                item.barrier(sycl::access::fence_space::local_space);
                const bool more = s + 2 < slabs;
                if (more) load((s + 2) * TileK);
                compute(1);
                if (more) store(0);
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (s < slabs) compute(0);  // an odd slab count ends in buffer 0, with no barrier after
        }

        // The NN kernel's epilogue: the 16-byte band store for a group inside C with 16-byte
        // columns, the WP-B staged store (lo = both A buffers, hi = both B; two passes at TileK = 8) for any other.
        const bool beta_zero = wide_scalar::dev_is_zero(beta);
        const bool c_vec = a_in && b_in && ldc % VecN == 0 && reinterpret_cast<std::uintptr_t>(Cb) % 16u == 0;
        if (!c_vec) {
            if (kPipe || slabs % 2 == 1) item.barrier(sycl::access::fence_space::local_space);
            wide_scalar::staged_c_store<TileM, TileN, Threads, 2 * TileK * TileM, 2 * TileK * TileN, EpiBatch>(
                item, reinterpret_cast<D*>(sa), reinterpret_cast<D*>(sb), Cb, ldc, m0, n0, m, n, tid, alpha, beta,
                beta_zero, [&](int c0, int cw, auto slot) {
BATCHLAS_UNROLL_FULL
                    for (int bm = 0; bm < NBM; ++bm) {
BATCHLAS_UNROLL_FULL
                        for (int j = 0; j < TTN; ++j) {
                            const int c = (j / VecN) * NSep + tx * VecN + j % VecN - c0;
                            if (c < 0 || c >= cw) continue;
                            wide_scalar::Vec16<D> v;
BATCHLAS_UNROLL_FULL
                            for (int w = 0; w < VecN; ++w) v.v[w] = accum[bm * VecN + w][j];
                            wide_scalar::vec_ref(slot(c, bm * MSep + ty * VecN)) = v;
                        }
                    }
                });
            return;
        }
BATCHLAS_UNROLL_FULL
        for (int bm = 0; bm < NBM; ++bm) {
            const int gm = m0 + bm * MSep + ty * VecN;
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TTN; ++j) {
                const int gn = n0 + (j / VecN) * NSep + tx * VecN + j % VecN;
                D* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                wide_scalar::Vec16<D> out;
                wide_scalar::Vec16<D> prior{};
                if (!beta_zero) prior = wide_scalar::vec_ref(const_cast<const D*>(p));
BATCHLAS_UNROLL_FULL
                for (int w = 0; w < VecN; ++w)
                    out.v[w] = wide_scalar::lin_epi(alpha, beta_zero ? D{} : beta, accum[bm * VecN + w][j], prior.v[w]);
                wide_scalar::vec_ref(p) = out;
            }
        }
    }
};
#undef BATCHLAS_WIDE_V2_REQD_WG

// The leg predicate: may every A and B load be one 16-byte move? Bases 16-byte aligned, ld and
// batch stride whole granules. Never a routing term; C's band store is decided per group.
template <typename T>
inline bool wide64_v2_vec_loads(const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B) {
    constexpr int VecN = 16 / static_cast<int>(sizeof(T));
    auto aligned = [](const T* p, int64_t ld, int64_t stride) {
        return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % 16u) == 0 && (ld % VecN) == 0 &&
            (stride % VecN) == 0;
    };
    return aligned(A.data_ptr(), A.ld(), A.stride()) && aligned(B.data_ptr(), B.ld(), B.stride());
}

template <typename T, bool VecLoads, int OA, int OB, int TileK, int TileM, int TileN, int Stages>
class GemmWideV2TileKernel;

template <typename T, Transpose OpA, Transpose OpB, int TileK, bool VecLoads, int TileM = 64, int TileN = 64,
          int Stages = 2>
Event launch_wide64_v2_leg(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha, T beta, const char* trace) {
    BATCHLAS_KERNEL_TRACE_SCOPE(trace);
    using Body = GemmWide64Body<T, OpA, OpB, VecLoads, TileK, TileM, TileN, Stages>;
    constexpr int oa = static_cast<int>(OpA), ob = static_cast<int>(OpB);
    using Name = std::conditional_t<TileM == 64 && TileN == 64, GemmRegister64x64K16WideKernel<T, VecLoads, oa, ob, TileK>,
                                    GemmWideV2TileKernel<T, VecLoads, oa, ob, TileK, TileM, TileN, Stages>>;
    using D = typename Body::D;
    const int m = static_cast<int>(Body::KA ? A.cols() : A.rows());
    const int k = static_cast<int>(Body::KA ? A.rows() : A.cols());
    const int n = static_cast<int>(Body::KB ? B.cols() : B.rows());
    const sycl::range<3> local(1, 1, Body::Threads);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()), static_cast<size_t>((m + Body::TileM - 1) / Body::TileM),
                                static_cast<size_t>((n + Body::TileN - 1) / Body::TileN) * Body::Threads);
    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<wide_scalar::Vec16<float>, 1> tile_a(sycl::range<1>(Body::Stages * Body::BufA / 4), h);
        sycl::local_accessor<wide_scalar::Vec16<float>, 1> tile_b(sycl::range<1>(Body::Stages * Body::BufB / 4), h);
        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));
        Body body{reinterpret_cast<const float*>(A.data_ptr()), reinterpret_cast<const float*>(B.data_ptr()),
                  reinterpret_cast<D*>(C.data_ptr()), m, n, k, static_cast<int>(A.ld()), static_cast<int>(B.ld()),
                  static_cast<int>(C.ld()), A.stride(), B.stride(), C.stride(), static_cast<int>(A.batch_size()),
                  alpha_d, beta_d, tile_a, tile_b};
        batchlas::impl::parallel_for<Name>(h, sycl::nd_range<3>(global, local), body);
    });
    return ctx.get_event();
}

// trace names the element-load leg, trace_vec the 16-byte one; the _s3 pair the 128x128 cp.async legs.
template <typename T, Transpose OpA, Transpose OpB, int TileK, int TileM = 64, int TileN = 64>
Event launch_wide64_v2(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C,
                       T alpha, T beta, const char* trace, const char* trace_vec, const char* trace_s3 = nullptr,
                       const char* trace_vec_s3 = nullptr) {
    if constexpr (TileM == 128 && TileN == 128) {
        // The leg (WP-V): three cp.async stages where DPC++ emits cp.async, the device launches
        // their 48 KiB and both bases are 8-byte aligned; measured on sm_120 only, so sm_80-90 take
        // it only under BATCHLAS_GEMM_WIDE_STAGES=3 (=2 forces the register prefetch everywhere).
        // Else the register prefetch (two slabs, can_run's wide_slm_bytes term).
        // evidence: docs/perf/gemm.md#gemm-wp-v-three-copy-async-stages-for-the-128x128-complex-float-tile
        using Pipe = GemmWide64Body<T, OpA, OpB, true, TileK, TileM, TileN, 3>;
        constexpr std::size_t pipe_bytes = Pipe::Stages * (Pipe::BufA + Pipe::BufB) * sizeof(float);
        const int forced = batchlas::settings().geometry.gemm_wide_stages;
        auto at8 = [](const T* p) { return reinterpret_cast<std::uintptr_t>(p) % 8u == 0; };
        if (!BATCHLAS_SYCL_IMPL_ACPP && forced != 2 &&
            ctx.device().cuda_compute_capability() >= (forced == 3 ? 80 : 120) && at8(A.data_ptr()) &&
            at8(B.data_ptr()) &&
            resident::device_slm_budget(impl::local_mem_bytes(ctx->get_device(), Pipe::Threads)) >= pipe_bytes) {
            if (wide64_v2_vec_loads<T>(A, B))
                return launch_wide64_v2_leg<T, OpA, OpB, TileK, true, TileM, TileN, 3>(ctx, A, B, C, alpha, beta,
                                                                                      trace_vec_s3);
            return launch_wide64_v2_leg<T, OpA, OpB, TileK, false, TileM, TileN, 3>(ctx, A, B, C, alpha, beta,
                                                                                   trace_s3);
        }
    }
    if (wide64_v2_vec_loads<T>(A, B))
        return launch_wide64_v2_leg<T, OpA, OpB, TileK, true, TileM, TileN>(ctx, A, B, C, alpha, beta, trace_vec);
    return launch_wide64_v2_leg<T, OpA, OpB, TileK, false, TileM, TileN>(ctx, A, B, C, alpha, beta, trace);
}

}  // namespace batchlas::sycl_gemm

#pragma once

// A 64x64x16 register-tiled GEMM with a 4x4 thread tile. The 4x4 tile is the widest
// shape that stays inside the 65,536 registers-per-block limit -- the 128x128 kernel's 8x8
// tile throws at launch for complex<double>. Two kernels: the NN one (double and
// complex<double>; register_wide_transposed.hh serves their other forms) and wide v2 below,
// every form for float and complex<float>.
// evidence: docs/perf/gemm.md#the-wide-scalar-kernel

#include <src/sycl/device_scalar.hh>
#include <src/ops/gemm/gemm_kernels.hh>
#include <src/ops/gemm/kernels/vec4.hh>

#include <src/linalg-impl.hh>
#include <src/sycl/kernel_attrs.hh>

#include <batchlas/backend_config.h>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::sycl_gemm {

namespace wide_scalar {

// std::complex is deliberately kept out of device code; see ../device_scalar.hh.
using batchlas::sycl_device::Cx;
using batchlas::sycl_device::DevMap;
using batchlas::sycl_device::dev_is_zero;
using batchlas::sycl_device::fma_acc;

inline float lin_epi(float alpha, float beta, float acc, float prior) {
    return alpha * acc + beta * prior;
}
inline double lin_epi(double alpha, double beta, double acc, double prior) {
    return alpha * acc + beta * prior;
}
template <typename R>
inline Cx<R> lin_epi(Cx<R> alpha, Cx<R> beta, Cx<R> acc, Cx<R> prior) {
    Cx<R> o;
    o.re = alpha.re * acc.re - alpha.im * acc.im + beta.re * prior.re - beta.im * prior.im;
    o.im = alpha.re * acc.im + alpha.im * acc.re + beta.re * prior.im + beta.im * prior.re;
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
template <int TileM, int TileN, int Threads, int LoElems, int HiElems, typename D, typename Put>
inline void staged_c_store(const sycl::nd_item<3>& item, D* lo, D* hi, D* Cb, int ldc, int m0, int n0, int m,
                           int n, int tid, D alpha, D beta, bool beta_zero, Put&& put) {
    static_assert(LoElems % TileM == 0 && HiElems % TileM == 0, "a stage column never straddles lo and hi");
    constexpr int CW = (LoElems + HiElems) / TileM;
    constexpr int Passes = (TileN + CW - 1) / CW;
    constexpr int Batch = 2;  // prior loads in flight per thread
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

// ---- wide v2: every transpose form, float and complex<float> -----------------------------
//
// One 64x64 kernel per (op(A), op(B)) over {N, T, C} and slab depth 16 or 8. Staging is a pure
// copy into one shared layout, s[kk][i] in 16-byte granules (i = m for A, n for B); conjugation
// is folded into the FMA signs. Two buffers, one barrier per slab, the next slab prefetched into
// ext_vector registers. The leg (16-byte or element global loads) is per launch; whether a
// slab is predicated is per work-group: only an edge group or the k tail pays for bounds.
// evidence: docs/perf/gemm.md#gemm-wide-v2-one-double-buffered-64x64-kernel-for-every-form

template <typename T>
inline constexpr bool kWide64V2 = std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;

namespace wide_v2 {

using batchlas::sycl_device::Cx;

// The XOR applied to the granule index of row kk, identically on store and fragment read; CF
// floats per element. It makes the k-contiguous operand's transposing store conflict-free: one
// bank-conflict set (an 8-lane phase of a 16-byte store, a warp of float ones) spans 4 row groups
// (kk / VecN) x 2 adjacent granules (GemmWide64Body::first_k), so bits 1-2 follow the row
// group. Four values, so the fragment read needs only four offsets. Float at TileK = 8 has two
// row groups x 4 granules per warp, so its row group flips bit 2.
template <int CF, int TileK>
constexpr int swz(int kk) {
    if constexpr (CF == 1 && TileK == 8) return ((kk / 4) & 1) << 2;
    else return ((kk / (4 / CF)) & 3) << 1;
}

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
template <bool CA, bool CB>
inline void fma_signed(Cx<float>& acc, Cx<float> a, Cx<float> b) {
    acc.re = sycl::fma(a.re, b.re, acc.re);
    acc.re = sycl::fma(CA != CB ? a.im : -a.im, b.im, acc.re);
    acc.im = sycl::fma(CB ? -a.re : a.re, b.im, acc.im);
    acc.im = sycl::fma(CA ? -a.im : a.im, b.re, acc.im);
}

template <typename D>
inline D elem(const Vec4<float>& t, int w) {
    if constexpr (std::is_same_v<D, float>) return t[w];
    else return D{t[2 * w], t[2 * w + 1]};
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
BATCHLAS_UNROLL_FULL
                for (int e = 0; e < CF; ++e) r[c][q * CF + e] = ok ? p[(base + q) * CF + e] : 0.0f;
            }
        }
    }
}

// The registers into shared row kk at granule (i / VecN) ^ swz(kk). Along i, a register is
// one granule. Along k, complex<float> pairs its two columns into one granule per row;
// float stores four scalars, one per row.
template <bool KC, int CF, int KStep, int TileK>
inline void sstore(float* s, const Vec4<float> (&r)[CF], int kb, int ib) {
    constexpr int VecN = 4 / CF, RowF = 64 * CF;
    if constexpr (!KC) {
BATCHLAS_UNROLL_FULL
        for (int c = 0; c < CF; ++c) {
            const int kk = kb + c * KStep;
            st16(s + kk * RowF + (((ib / VecN) ^ swz<CF, TileK>(kk)) << 2), r[c]);
        }
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

}  // namespace wide_v2

// A FUNCTOR, not a lambda, only so the launch bound can be spelled. Two groups per CU for
// complex<float> (its prefetch registers do not fit three groups' 85-register cap).
// TileK = 8 halves the slab; each lane still stages CF granules, so warps 0-3 stage A and 4-7 B.
template <typename T, Transpose OpA, Transpose OpB, bool VecLoads, int TileK_>
struct GemmWide64Body {
    using D = typename wide_scalar::DevMap<T>::type;
    static constexpr int TileM = 64, TileN = 64, TileK = TileK_, TT = 4, Threads = 256;
    static constexpr int CF = static_cast<int>(sizeof(D) / sizeof(float));
    static constexpr int VecN = 4 / CF;
    static constexpr int NB = TT / VecN;                         // fragment granules per operand
    static constexpr int MSep = TileM / NB, NSep = TileN / NB;   // band separation
    static constexpr int RowF = TileM * CF, BufF = TileK * RowF;  // floats
    static constexpr int GPR = TileM / VecN;                     // granules per row
    static constexpr bool Split = TileK == 8;
    static constexpr int OpThreads = Split ? Threads / 2 : Threads;  // staging lanes per operand
    static constexpr int KStep = OpThreads / GPR;
    static constexpr bool KA = OpA != Transpose::NoTrans, KB = OpB == Transpose::NoTrans;
    static constexpr bool CA = CF == 2 && OpA == Transpose::ConjTrans;
    static constexpr bool CB = CF == 2 && OpB == Transpose::ConjTrans;
    static constexpr int kMinGroupsPerCu = CF == 1 ? 3 : 2;
    static_assert(kWide64V2<T> && TileM == TileN, "one staging map serves A and B");
    static_assert(TileK == 16 || TileK == 8, "first_k and swz map these two slab depths");
    static_assert(GPR * TileK == CF * OpThreads, "CF granules per staging lane per operand");

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
    // tid is the lane's index among its operand's OpThreads staging lanes.
    static constexpr int KG = TileK / VecN;  // row groups along k
    template <bool KC>
    static int first_k(int tid) {
        if constexpr (!KC) return tid / GPR;
        else if constexpr (CF == 1 || TileK == 8) return (tid % KG) * VecN;
        else return ((tid & 3) | ((tid >> 1) & 4)) * VecN;
    }
    template <bool KC>
    static int first_i(int tid) {
        if constexpr (!KC) return (tid % GPR) * VecN;
        else if constexpr (CF == 1 || TileK == 8) return (tid / KG) * CF;
        else return (((tid >> 2) & 1) | ((tid >> 3) & ~1)) * CF;
    }

    template <bool KC>
    __attribute__((always_inline)) static void fetch(Vec4<float> (&r)[CF], const float* p, int ld, int k0, int kb,
                                                     int ib, int k, int lim, bool full) {
        if (full) wide_v2::gload<false, VecLoads, KC, CF, KStep>(r, p, ld, k0, kb, ib, k, lim);
        else wide_v2::gload<true, VecLoads, KC, CF, KStep>(r, p, ld, k0, kb, ib, k, lim);
    }

    // always_inline: left out of line (complex<float>), the body spilled behind a call.
    BATCHLAS_LAUNCH_BOUNDS(Threads, kMinGroupsPerCu)
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

        D accum[TT][TT];
BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TT; ++i) {
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TT; ++j) accum[i][j] = D{};
        }

        // Split: one register set, holding A on an A lane and B on a B lane.
        Vec4<float> ra[CF], rb[CF];
        auto& rb_ = Split ? ra : rb;
        auto load = [&](int k0) __attribute__((always_inline)) {
            const bool kin = k0 + TileK <= k;
            if (stage_a) fetch<KA>(ra, pa, lda, k0, a_kb, a_ib, k, m - m0, a_in && kin);
            if (!Split || !stage_a) fetch<KB>(rb_, pb, ldb, k0, b_kb, b_ib, k, n - n0, b_in && kin);
        };
        auto store = [&](int buf) __attribute__((always_inline)) {
            if (stage_a) wide_v2::sstore<KA, CF, KStep, TileK>(sa + buf * BufF, ra, a_kb, a_ib);
            if (!Split || !stage_a) wide_v2::sstore<KB, CF, KStep, TileK>(sb + buf * BufF, rb_, b_kb, b_ib);
        };
        auto compute = [&](int buf) __attribute__((always_inline)) {
            const float* xa = sa + buf * BufF;
            const float* xb = sb + buf * BufF;
BATCHLAS_UNROLL_FULL
            for (int kk = 0; kk < TileK; ++kk) {
                D af[TT], bf[TT];
BATCHLAS_UNROLL_FULL
                for (int band = 0; band < NB; ++band) {
                    const Vec4<float> ta =
                        ld16(xa + kk * RowF + (((band * (GPR / NB) + ty) ^ swz<CF, TileK>(kk)) << 2));
                    const Vec4<float> tb =
                        ld16(xb + kk * RowF + (((band * (GPR / NB) + tx) ^ swz<CF, TileK>(kk)) << 2));
BATCHLAS_UNROLL_FULL
                    for (int w = 0; w < VecN; ++w) {
                        af[band * VecN + w] = elem<D>(ta, w);
                        bf[band * VecN + w] = elem<D>(tb, w);
                    }
                }
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < TT; ++i) {
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < TT; ++j) fma_signed<CA, CB>(accum[i][j], af[i], bf[j]);
                }
            }
        };

        // One barrier per slab, as in the 128x128 kernel: slab s+1 is stored into the buffer
        // slab s-1 was read from, and every read of it finished before the barrier ending s-1.
        const int slabs = (k + TileK - 1) / TileK;
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

        // The NN kernel's epilogue: the 16-byte band store for a group inside C with 16-byte
        // columns, the WP-B staged store (lo = both A buffers, hi = both B; two passes at TileK = 8) for any other.
        const bool beta_zero = wide_scalar::dev_is_zero(beta);
        const bool c_vec = a_in && b_in && ldc % VecN == 0 && reinterpret_cast<std::uintptr_t>(Cb) % 16u == 0;
        if (!c_vec) {
            if (slabs % 2 == 1) item.barrier(sycl::access::fence_space::local_space);
            wide_scalar::staged_c_store<TileM, TileN, Threads, 2 * TileK * TileM, 2 * TileK * TileN>(
                item, reinterpret_cast<D*>(sa), reinterpret_cast<D*>(sb), Cb, ldc, m0, n0, m, n, tid, alpha, beta,
                beta_zero, [&](int c0, int cw, auto slot) {
BATCHLAS_UNROLL_FULL
                    for (int bm = 0; bm < NB; ++bm) {
BATCHLAS_UNROLL_FULL
                        for (int j = 0; j < TT; ++j) {
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
        for (int bm = 0; bm < NB; ++bm) {
            const int gm = m0 + bm * MSep + ty * VecN;
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TT; ++j) {
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

template <typename T, Transpose OpA, Transpose OpB, int TileK, bool VecLoads>
Event launch_wide64_v2_leg(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha, T beta, const char* trace) {
    BATCHLAS_KERNEL_TRACE_SCOPE(trace);
    using Body = GemmWide64Body<T, OpA, OpB, VecLoads, TileK>;
    using D = typename Body::D;
    const int m = static_cast<int>(Body::KA ? A.cols() : A.rows());
    const int k = static_cast<int>(Body::KA ? A.rows() : A.cols());
    const int n = static_cast<int>(Body::KB ? B.cols() : B.rows());
    const sycl::range<3> local(1, 1, Body::Threads);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()), static_cast<size_t>((m + Body::TileM - 1) / Body::TileM),
                                static_cast<size_t>((n + Body::TileN - 1) / Body::TileN) * Body::Threads);
    ctx->submit([&](sycl::handler& h) {
        constexpr size_t granules = 2 * Body::BufF / 4;
        sycl::local_accessor<wide_scalar::Vec16<float>, 1> tile_a(sycl::range<1>(granules), h);
        sycl::local_accessor<wide_scalar::Vec16<float>, 1> tile_b(sycl::range<1>(granules), h);
        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));
        Body body{reinterpret_cast<const float*>(A.data_ptr()), reinterpret_cast<const float*>(B.data_ptr()),
                  reinterpret_cast<D*>(C.data_ptr()), m, n, k, static_cast<int>(A.ld()), static_cast<int>(B.ld()),
                  static_cast<int>(C.ld()), A.stride(), B.stride(), C.stride(), static_cast<int>(A.batch_size()),
                  alpha_d, beta_d, tile_a, tile_b};
        batchlas::impl::parallel_for<
            GemmRegister64x64K16WideKernel<T, VecLoads, static_cast<int>(OpA), static_cast<int>(OpB), TileK>>(
            h, sycl::nd_range<3>(global, local), body);
    });
    return ctx.get_event();
}

// trace names the element-load leg, trace_vec the 16-byte one.
template <typename T, Transpose OpA, Transpose OpB, int TileK>
Event launch_wide64_v2(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C,
                       T alpha, T beta, const char* trace, const char* trace_vec) {
    if (wide64_v2_vec_loads<T>(A, B))
        return launch_wide64_v2_leg<T, OpA, OpB, TileK, true>(ctx, A, B, C, alpha, beta, trace_vec);
    return launch_wide64_v2_leg<T, OpA, OpB, TileK, false>(ctx, A, B, C, alpha, beta, trace);
}

}  // namespace batchlas::sycl_gemm

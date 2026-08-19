#pragma once

// A 64x64x16 register-tiled GEMM with a 4x4 thread tile, for WIDE SCALARS.
//
// This is the only register-tiled variant in this directory that serves a
// scalar other than float. Everything else here is reachable only from the
// `if constexpr (std::is_same_v<T, float>)` ladder in select_kernel_variant,
// so double, complex<float> and complex<double> fall straight through it to
// Tiled16 -- one accumulator per thread, std::complex operator* in the inner
// loop, and a scattered epilogue.
//
// Measured on RTX 4090 / sm_89 at 256^3 b512, 512^3 b128 and 1024^3 b32, at
// beta 0 and beta 1, against a faithful standalone replica of Tiled16 and
// against cuBLAS:
//
//   complex<float>  : 7.0-7.7x Tiled16, 0.98-1.08x cuBLAS CGEMM
//   complex<double> : 3.56-3.60x Tiled16, 1.12x cuBLAS ZGEMM
//   double          : 1.01-1.08x Tiled16, 1.07-1.15x cuBLAS DGEMM
//   float           : 0.85-0.93x cuBLAS SGEMM -- WORSE than the in-tree
//                     128x128 kernel, which is why float never routes here.
//
// See WP2_WIDE_SCALAR_GEMM_VERDICT.md and experiments/wide_scalar_gemm/.
//
// Read the `double` row as small on purpose. FP64 on a 4090 is 1/64 of FP32,
// so the ceiling at the observed clocks is ~1.44 TFLOP/s; this kernel reaches
// 1.415 (99% of it) but the naive Tiled16 already reaches 1.33 (92%). There
// was never 3x on the table for double on this part and no tile design can
// find it. That conclusion is 4090-specific and INVERTS on a 1:2-FP64
// datacenter part, where Tiled16 would not be near the ceiling.
//
// WHY THIS SHAPE, AND WHY NOT THE 128x128 ONE
//
// A 4x4 thread tile is 16 accumulators, which is 32 registers for double and
// complex<float> and 64 for complex<double>. The 8x8 tile that the float
// 128x128 kernel uses would be 128 and 256 respectively; measured, that is
// still launchable for double and complex<float> (208 and 247 registers, zero
// spill) but for complex<double> it exceeds the 65,536 registers-per-block
// limit and throws at launch. This kernel uses ONE tile shape for all four
// scalars and is the only candidate with no unlaunchable and no spilling
// configuration: 72-134 registers, zero spill bytes, on all four.
//
// THE FIVE LOAD-BEARING DETAILS
//
// Each was found by reading PTX, and dropping any one of them reverts a
// measured property. They are called out again at their sites below.
//
//   1. The access granule is 16 BYTES, not 4 elements. Packet4<T> from
//      register_128x128.hh is `alignas(4 * sizeof(T))` -- 32 bytes for double
//      and 64 for complex<double>, load forms that do not exist in SASS. See
//      Vec16 below.
//   2. __attribute__((may_alias)) on the punning types, or -O3 reorders the
//      shared stores against the fragment loads across the barrier.
//   3. The whole-granule staging copy must be a native LLVM vector type, not
//      a struct copy -- SROA splits the latter back into element accesses.
//   4. std::complex must never reach device code: re-typed to POD Cx<R> at
//      the pointer boundary, with the multiply written out as four fma()s.
//   5. Shared strides exactly TileM / TileN, and m fastest-varying in the
//      epilogue.

#include "../device_scalar.hh"
#include "../gemm_kernels.hh"

#include "../../linalg-impl.hh"

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::sycl_gemm {

namespace wide_scalar {

// ---------------------------------------------------------------------------
// (4) Device scalar types
// ---------------------------------------------------------------------------
//
// LIFTED to src/sycl/device_scalar.hh when TRSM needed the same types. They are
// aliased back into this namespace so every use below is unchanged; the move was
// verified with scripts/register_probe.sh, which still reports 56 / 76 / 80 /
// 132 registers and zero spill for the four instantiations of this kernel.
//
// std::complex is still deliberately kept out of device code -- see the note in
// that header. lin_epi stays here because it is this kernel's epilogue, not a
// general scalar operation.

using batchlas::sycl_device::Cx;
using batchlas::sycl_device::DevMap;
using batchlas::sycl_device::dev_is_zero;
using batchlas::sycl_device::fma_acc;

// alpha * acc + beta * prior. Stays here: it is this kernel's epilogue form,
// not a general scalar operation, and it is where the beta == 0 branch that
// once cost 15 TFLOP/s lives.
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

// ---------------------------------------------------------------------------
// (1) + (2) The 16-byte access granule
// ---------------------------------------------------------------------------
//
// Every vectorized load and store in this kernel is EXACTLY 16 bytes, whatever
// the scalar width. That is what keeps an 8-lane LDS phase on exactly the 32
// banks. It is why this is not Packet4<T>: for double a 4-element packet is 32
// bytes and its 8-lane phase straddles 256 bytes -- a 2-way bank conflict --
// while for complex<double> it is 64 bytes and 4-way.
//
// may_alias, because these types read and write objects of type D, which is a
// strict-aliasing violation without it. -O3 is entitled to reorder the shared
// stores against the fragment loads if it believes they cannot alias, and the
// barrier alone does not stop that.
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

// (3) A raw 16-byte move, for staging -- a pure bit copy, so the scalar type
// is irrelevant. This has to be a native LLVM vector type rather than a
// struct: a `struct { T v[N]; }` copy is split by SROA into element loads and
// stores and the 16-byte form is lost. Measured, in the PTX, as four
// ld.global.b64 + four st.shared.b64 where two ld.global.v2.b64 + two
// st.shared.v2.b64 were intended. `struct` is fine for the fragment *loads*,
// which are consumed element-wise and stay vectorized; it is only the
// whole-granule copy that needs this.
typedef double Raw16Base __attribute__((ext_vector_type(2)));
typedef Raw16Base Raw16 __attribute__((may_alias));

inline Raw16 raw16_load(const void* p) { return *reinterpret_cast<const Raw16*>(p); }
inline void raw16_store(void* p, Raw16 v) { *reinterpret_cast<Raw16*>(p) = v; }

// ---------------------------------------------------------------------------
// Tile geometry
// ---------------------------------------------------------------------------

struct Tile {
    static constexpr int M = 64;
    static constexpr int N = 64;
    static constexpr int K = 16;
    static constexpr int TT = 4;                           // thread tile, both directions
    static constexpr int LocalRows = M / TT;               // 16, the m direction
    static constexpr int LocalCols = N / TT;               // 16, the n direction
    static constexpr int Threads = LocalRows * LocalCols;  // 256
    // (5) No padding. An aligned stride is what lets the fragment loads
    // vectorize; an odd stride means the compiler can never prove 16-byte
    // alignment and every fragment load degrades to a scalar ld.shared.
    static constexpr int AStride = M;
    static constexpr int BStride = N;
    static constexpr int PerThreadA = M * K / Threads;  // 4
    static constexpr int PerThreadB = K * N / Threads;  // 4
};

static_assert(Tile::M * Tile::K == Tile::PerThreadA * Tile::Threads, "A staging");
static_assert(Tile::K * Tile::N == Tile::PerThreadB * Tile::Threads, "B staging");

}  // namespace wide_scalar

template <typename T, bool AlignedFastPath>
class GemmRegister64x64K16WideKernel;

// Does this problem satisfy everything the unpredicated path assumes?
//
// Note the granule: 16 BYTES, not 4 elements. VecLen is 4/2/2/1 for
// float/double/complex<float>/complex<double>, so the byte width of every
// vector access is pinned at 128 bits for every scalar.
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

// NN only. The kernel reads A as m x k and B as k x n directly, so it cannot
// serve a transposed operand; the caller falls back rather than transposing.
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

    // (4) The whole kernel runs on the POD device type. std::complex is
    // re-typed here, at the pointer boundary, and never crosses into the
    // kernel body -- including alpha and beta, which are reinterpreted
    // exactly as the operand pointers are.
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

        h.parallel_for<GemmRegister64x64K16WideKernel<T, AlignedFastPath>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch) {
                    return;
                }

                // (5) m is the fastest-varying thread index. C is
                // column-major, so lanes that differ in m touch consecutive
                // addresses; lanes that differ in n would stride by ldc.
                // Getting this backwards is nearly free at beta == 0 and
                // catastrophic at beta != 0, because the read of C becomes one
                // transaction per lane.
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
#pragma unroll
                for (int i = 0; i < TT; ++i) {
#pragma unroll
                    for (int j = 0; j < TT; ++j) {
                        accum[i][j] = D{};
                    }
                }

                // Staging coordinates.
                //
                //  A is column-major m x k, contiguous down m. The tile is cut
                //  into 16-byte granules and granule ids are handed out in
                //  linear tid order, one chunk at a time. Consecutive lanes
                //  therefore get consecutive granules: the global read of a
                //  warp is 32 x 16 = 512 contiguous bytes, and the shared
                //  write of an 8-lane phase is 128 contiguous bytes = exactly
                //  the 32 banks, conflict-free, for every scalar width.
                //
                //  (Handing each thread PerThreadA *consecutive elements* --
                //  what the float kernels do, where 4 floats happen to BE one
                //  granule -- would put lanes 32 bytes apart for double and 64
                //  for complex<double>, a 4- and 8-way shared-store conflict.)
                int a_gm[Chunks];
                int a_gk[Chunks];
#pragma unroll
                for (int c = 0; c < Chunks; ++c) {
                    const int g = tid + c * Tile::Threads;
                    a_gm[c] = (g % AGranPerRow) * VecN;
                    a_gk[c] = g / AGranPerRow;
                }

                //  B is column-major k x n, contiguous down k: a thread takes
                //  PerThreadB consecutive k from one n-column and scatters
                //  them into shared so that shared ends up [k][n]. The scatter
                //  is irreducibly non-vector on the store side (that is the
                //  transpose), and it costs a 4-way shared-bank conflict; it
                //  is 4 store instructions per k0 tile against 64 fragment
                //  loads, so it is left alone in favour of keeping the global
                //  read of B coalesced (4 lanes per column, 8 columns/warp).
                const int b_k = (tid % (TileK / Tile::PerThreadB)) * Tile::PerThreadB;
                const int b_n = tid / (TileK / Tile::PerThreadB);

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    if constexpr (AlignedFastPath) {
                        // ---- A: 16-byte global load -> 16-byte shared store
#pragma unroll
                        for (int c = 0; c < Chunks; ++c) {
                            raw16_store(
                                &sa[a_gk[c] * AStride + a_gm[c]],
                                raw16_load(Ab + (m0 + a_gm[c]) +
                                           static_cast<std::ptrdiff_t>(k0 + a_gk[c]) * lda));
                        }
                        // ---- B: vectorized load, scattered shared store.
                        // Same raw-granule trick as A on the load side; the
                        // scatter into [k][n] is what forces the store side to
                        // stay scalar.
                        alignas(16) D vb[Tile::PerThreadB];
#pragma unroll
                        for (int c = 0; c < Chunks; ++c) {
                            raw16_store(&vb[c * VecN],
                                        raw16_load(Bb + (k0 + b_k + c * VecN) +
                                                   static_cast<std::ptrdiff_t>(n0 + b_n) * ldb));
                        }
#pragma unroll
                        for (int i = 0; i < Tile::PerThreadB; ++i) {
                            sb[(b_k + i) * BStride + b_n] = vb[i];
                        }
                    } else {
                        // Predicated staging. The tile is always filled to its
                        // full 64x16, zero outside the matrix, so the inner
                        // loop below needs no bounds checks at all and is
                        // bit-identical between the two paths.
#pragma unroll
                        for (int c = 0; c < Chunks; ++c) {
                            const int gk_a = k0 + a_gk[c];
#pragma unroll
                            for (int e = 0; e < VecN; ++e) {
                                const int gm = m0 + a_gm[c] + e;
                                sa[a_gk[c] * AStride + a_gm[c] + e] =
                                    (gm < m && gk_a < k)
                                        ? Ab[gm + static_cast<std::ptrdiff_t>(gk_a) * lda]
                                        : D{};
                            }
                        }
                        const int gn_b = n0 + b_n;
#pragma unroll
                        for (int i = 0; i < Tile::PerThreadB; ++i) {
                            const int gk = k0 + b_k + i;
                            sb[(b_k + i) * BStride + b_n] =
                                (gk < k && gn_b < n)
                                    ? Bb[gk + static_cast<std::ptrdiff_t>(gn_b) * ldb]
                                    : D{};
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
                    for (int kk = 0; kk < TileK; ++kk) {
                        D af[TT];
                        D bf[TT];
#pragma unroll
                        for (int band = 0; band < NB; ++band) {
                            const Vec16<D> t =
                                vec_ref(&sa[kk * AStride + band * MSep + ty * Wb]);
#pragma unroll
                            for (int w = 0; w < Wb; ++w) {
                                af[band * Wb + w] = t.v[w];
                            }
                        }
#pragma unroll
                        for (int band = 0; band < NB; ++band) {
                            const Vec16<D> t =
                                vec_ref(&sb[kk * BStride + band * NSep + tx * Wb]);
#pragma unroll
                            for (int w = 0; w < Wb; ++w) {
                                bf[band * Wb + w] = t.v[w];
                            }
                        }
#pragma unroll
                        for (int i = 0; i < TT; ++i) {
#pragma unroll
                            for (int j = 0; j < TT; ++j) {
                                fma_acc(accum[i][j], af[i], bf[j]);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // ---- Epilogue ------------------------------------------
                // Within a band the Wb rows are consecutive in m, which is the
                // contiguous direction of a column-major C, so a whole band is
                // one 16-byte access. Lanes differ in ty, i.e. in m, so the
                // band bases are consecutive too.
                const bool beta_zero = dev_is_zero(beta_d);
#pragma unroll
                for (int bm = 0; bm < NB; ++bm) {
                    const int gm = m0 + bm * MSep + ty * Wb;
#pragma unroll
                    for (int j = 0; j < TT; ++j) {
                        const int bn = j / Wb;
                        const int w_n = j % Wb;
                        const int gn = n0 + bn * NSep + tx * Wb + w_n;
                        if constexpr (AlignedFastPath) {
                            D* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                            Vec16<D> out;
                            if (beta_zero) {
#pragma unroll
                                for (int w = 0; w < Wb; ++w) {
                                    out.v[w] =
                                        lin_epi(alpha_d, D{}, accum[bm * Wb + w][j], D{});
                                }
                            } else {
                                const Vec16<D> prior = vec_ref(const_cast<const D*>(p));
#pragma unroll
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
#pragma unroll
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

}  // namespace batchlas::sycl_gemm

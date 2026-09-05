#pragma once

// A 64x64x16 register-tiled GEMM with a 4x4 thread tile, for WIDE SCALARS:
// double, complex<float> and complex<double>. Float never routes here -- the
// in-tree 128x128 kernel is faster for it.
//
// The 4x4 tile is the only shape that is launchable and spill-free for all
// four scalars; the 128x128 kernel's 8x8 tile exceeds the 65,536
// registers-per-block limit for complex<double> and throws at launch.
//
// evidence: docs/perf/gemm.md#the-wide-scalar-kernel

#include "../device_scalar.hh"
#include "../gemm_kernels.hh"

#include "../../linalg-impl.hh"

#include <sycl/sycl.hpp>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::sycl_gemm {

namespace wide_scalar {

// Device scalar types. std::complex is deliberately kept out of device code;
// see src/sycl/device_scalar.hh.
using batchlas::sycl_device::Cx;
using batchlas::sycl_device::DevMap;
using batchlas::sycl_device::dev_is_zero;
using batchlas::sycl_device::fma_acc;

// alpha * acc + beta * prior -- this kernel's epilogue form.
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
// that is what keeps an 8-lane LDS phase on exactly the 32 banks. Packet4<T>
// would be 32 bytes for double and 64 for complex<double>, and conflict.
//
// may_alias is load-bearing: these types read and write objects of type D, and
// without it -O3 may reorder the shared stores against the fragment loads
// across the barrier.
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
// stores and the 16-byte form is lost. Only the whole-granule copy needs this;
// the fragment loads are consumed element-wise and stay vectorized as structs.
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

template <typename T, bool AlignedFastPath>
class GemmRegister64x64K16WideKernel;

// Does this problem satisfy everything the unpredicated path assumes? VecLen
// counts elements, but the granule it enforces is 16 bytes for every scalar.
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

    // The whole kernel runs on the POD device type: std::complex is re-typed
    // here, at the pointer boundary, and never crosses into the kernel body --
    // including alpha and beta.
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

                // m is the fastest-varying thread index: C is column-major, so
                // lanes differing in m touch consecutive addresses. Reversing
                // this is nearly free at beta == 0 and catastrophic otherwise,
                // because the read of C becomes one transaction per lane.
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

                // A is column-major m x k. Granule ids are handed out in
                // linear tid order, so consecutive lanes get consecutive
                // 16-byte granules and both the global read and the shared
                // write stay conflict-free at every scalar width. Handing each
                // thread PerThreadA consecutive *elements*, as the float
                // kernels do, is a 4- to 8-way shared-store conflict here.
                int a_gm[Chunks];
                int a_gk[Chunks];
#pragma unroll
                for (int c = 0; c < Chunks; ++c) {
                    const int g = tid + c * Tile::Threads;
                    a_gm[c] = (g % AGranPerRow) * VecN;
                    a_gk[c] = g / AGranPerRow;
                }

                // B is scattered into shared as [k][n]. The store side is
                // irreducibly non-vector and costs a 4-way bank conflict; that
                // is deliberate, to keep the global read of B coalesced.
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
                        // ---- B: vectorized load, scattered shared store
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
                        // Predicated staging always fills the tile to its full
                        // 64x16, zero outside the matrix, so the inner loop
                        // below needs no bounds checks and is bit-identical
                        // between the two paths.
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
                // Within a band the Wb rows are consecutive in m, so a whole
                // band is one 16-byte access into column-major C.
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

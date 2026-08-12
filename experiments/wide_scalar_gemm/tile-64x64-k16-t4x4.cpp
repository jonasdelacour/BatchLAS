// tile-64x64-k16-t4x4.cpp
//
// A batched register-tiled GEMM for WIDE SCALAR TYPES (double, complex<float>,
// complex<double>), modelled on src/sycl/gemm/register_128x128.hh but with a
// tile shape that fits in registers for scalars wider than 4 bytes.
//
// ---------------------------------------------------------------------------
// WHY 64x64x16 WITH A 4x4 THREAD TILE
// ---------------------------------------------------------------------------
// The float kernel's 8x8 thread tile is 64 accumulators. That is 64 registers
// for float, but 128 for double / complex<float> and 256 for complex<double> --
// past the 255-register hardware limit, so it spills. A 4x4 thread tile is 16
// accumulators:
//
//     float             16 registers of accumulator
//     double            32
//     complex<float>    32
//     complex<double>   64
//
// which leaves room for the fragments, the addressing, and the epilogue in
// every case. 256 threads (16x16) x 4x4 gives the 64x64 macro tile; K is
// deepened to 16 so that the staging still works out to exactly 4 elements per
// thread per tile (64*16 = 16*64 = 4*256), which is what keeps the global
// loads vectorized and the shared stores one-per-thread.
//
// Shared footprint is 2 * 16 * 64 = 2048 elements: 16 KB for double and
// complex<float>, 32 KB for complex<double>. All fit sm_89's 48 KB static
// limit with room for 3-6 resident blocks.
//
// ---------------------------------------------------------------------------
// WHAT IS PRESERVED FROM THE FLOAT KERNEL
// ---------------------------------------------------------------------------
//  1. FFMA-to-shared-load ratio. Per k-step a thread issues 2*4 = 8 scalar
//     elements of fragment and 16 scalar-type FMAs. Because every shared
//     access is a 16-byte granule, the fragment costs 2*TT/VecN LDS.128
//     instructions, where VecN = 16/sizeof(T):
//
//        T                 VecN  LDS.128/kstep  FFMA/kstep   ratio
//        float               4         2            16         8:1
//        double              2         4            16         4:1   (DFMA)
//        complex<float>      2         4            64        16:1
//        complex<double>     1         8            64         8:1   (DFMA)
//
//     complex<float> lands on exactly the 16:1 of the 128x128 float kernel,
//     because a complex FMA is four real FFMAs -- the arithmetic intensity of
//     complex arithmetic pays for the smaller tile. The two double rows are
//     lower in instruction terms, but the FP64 pipe on a consumer Ada part is
//     1/64 the FP32 rate, so the shared pipe has 64x more headroom to spare
//     there; see the honesty note at the bottom of this comment.
//
//  2. Shared tiles stored with stride EXACTLY TileM / TileN. No +1 padding:
//     an odd stride would make 16-byte alignment unprovable and every fragment
//     load would degrade to a scalar ld.shared.
//
//  3. B staged as [k][n], not [n][k], so a thread's fragment is contiguous in
//     n and vectorizes.
//
//  4. Bank-conflict-free vectorized loads. The float kernel splits its 8 rows
//     into two 4-wide bands so that 8 lanes x 16 bytes covers exactly the 32
//     banks. Generalized here: the band width is Wb = 16/sizeof(T) elements --
//     i.e. ALWAYS 16 bytes -- and the thread tile splits into NB = TT/Wb bands
//     separated by TileM/NB:
//
//        float            Wb=4, NB=1, rows {ty*4 .. +3}
//        double           Wb=2, NB=2, rows {ty*2,+1} u {32+ty*2,+1}
//        complex<float>   Wb=2, NB=2, same
//        complex<double>  Wb=1, NB=4, rows {ty, 16+ty, 32+ty, 48+ty}
//
//     In every case an 8-lane phase of an LDS.128 covers 8 * 16 = 128
//     contiguous bytes = exactly the 32 banks, one distinct bank per word.
//     A 4-element packet (which is what Packet4<T> would give) would be 32 or
//     64 bytes wide for these types and its 8-lane phase would straddle 256 or
//     512 bytes -- a 2- or 4-way bank conflict. That is the one place this
//     kernel deliberately diverges from register_128x128.hh, and it is forced
//     by the scalar width.
//
//     The same reasoning applies to the A staging, and there it forces a
//     second divergence: the float kernel hands each thread 4 CONSECUTIVE m,
//     which for float is exactly one granule but for double is 32 bytes and
//     for complex<double> 64, so the shared-store phase would straddle and
//     conflict 4- or 8-way. This kernel hands out 16-byte GRANULES in linear
//     tid order instead, which is identical to the float kernel when VecN == 4
//     and conflict-free for every narrower VecN.
//
//     Verified in the emitted PTX (see inspect.sh): every shared fragment load
//     is a 16-byte ld.shared.v2.b64 / ld.shared.v4.b32, the A staging is
//     ld.global.v2 -> st.shared.v2, and there are no scalar global accesses on
//     the fast path at all. The one deliberately scalar store is B's k->n
//     scatter, which no layout can vectorize.
//
//  5. The m index is the FASTEST-VARYING thread index (ty = local_id(2)), so
//     the epilogue's read and write of a column-major C are coalesced. The
//     float kernel measured 26.0 -> 41.1 TFLOP/s on this one change, visible
//     only at beta != 0. This harness therefore DEFAULTS TO beta = 1.
//
//  6. The complex multiply is written out explicitly as four sycl::fma calls.
//     std::complex operator* emits an isnan branch and a call to __mulsc3 in
//     device code; std::complex never enters device code here at all -- the
//     launcher reinterprets to a plain aggregate Cx<R>.
//
//  7. The accumulator is a plain local array in the kernel body. Nothing is
//     passed by reference out of the inner loop.
//
// ---------------------------------------------------------------------------
// HONEST ASSESSMENT OF THE RATIO
// ---------------------------------------------------------------------------
// For complex<float> the 16:1 ratio is the same as the float kernel's and this
// shape should be competitive on its own terms.
//
// For double the instruction ratio is 4:1, which would be far too low if DFMA
// ran at FP32 rate. It does not on a 4090: FP64 is 1/64 of FP32, i.e. 2
// DFMA/clk/SM against 128 FFMA/clk/SM, while the shared pipe is unchanged.
// So 4:1 against DFMA has roughly 16x more headroom than 8:1 against FFMA
// does. On a datacenter part with 1:2 FP64 (A100/H100) this shape WOULD be
// shared-bound and the fix is a 4x8 or 8x4 thread tile (32 accumulators = 64
// registers for double, still safe) which doubles the ratio to 8:1 at the cost
// of a 128x64 or 64x128 macro tile; that is the next step if this is ever
// retargeted off consumer Ada.
//
// For complex<double> the 8:1 is fine for the same reason as double.
//
// ---------------------------------------------------------------------------
// BUILD
// ---------------------------------------------------------------------------
//   /opt/dpcpp-cuda/bin/clang++ -O3 -std=c++20 -fsycl \
//       -fsycl-targets=nvidia_gpu_sm_89 --cuda-path=/usr/local/cuda-13.2 \
//       -Xcuda-ptxas -v tile-64x64-k16-t4x4.cpp -o tile-64x64-k16-t4x4
//
// RUN
//   ./tile-64x64-k16-t4x4 --dtype double --m 512 --n 512 --k 512 \
//       --batch 64 --beta 1
//
// Prints exactly one machine-readable line:
//   RESULT dtype=.. m=.. n=.. k=.. batch=.. beta=.. tile=.. ms=.. tflops=.. maxrelerr=..

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace wsg {

// ===========================================================================
// Device scalar types
// ===========================================================================

// A plain aggregate complex. std::complex is deliberately kept out of device
// code: its operator* is Annex-G conformant, which means an isnan branch and a
// call to __mulsc3 / __muldc3 in the inner loop.
template <typename R>
struct Cx {
    R re;
    R im;
};

template <typename T>
struct DevMap {
    using type = T;
    using real = T;
    static constexpr bool is_complex = false;
};

template <typename R>
struct DevMap<std::complex<R>> {
    using type = Cx<R>;
    using real = R;
    static constexpr bool is_complex = true;
};

static_assert(sizeof(Cx<float>) == sizeof(std::complex<float>), "layout");
static_assert(sizeof(Cx<double>) == sizeof(std::complex<double>), "layout");

// --- zero / compare -------------------------------------------------------

template <typename R>
inline bool dev_is_zero(R x) {
    return x == R(0);
}
template <typename R>
inline bool dev_is_zero(Cx<R> x) {
    return x.re == R(0) && x.im == R(0);
}

// --- the multiply-accumulate, written out ---------------------------------

// Real: one FMA.
inline void fma_acc(float& acc, float a, float b) { acc = sycl::fma(a, b, acc); }
inline void fma_acc(double& acc, double a, double b) { acc = sycl::fma(a, b, acc); }

// Complex: four FMAs, no branches, no libcall. This is the whole point.
template <typename R>
inline void fma_acc(Cx<R>& acc, Cx<R> a, Cx<R> b) {
    acc.re = sycl::fma(a.re, b.re, acc.re);
    acc.re = sycl::fma(-a.im, b.im, acc.re);
    acc.im = sycl::fma(a.re, b.im, acc.im);
    acc.im = sycl::fma(a.im, b.re, acc.im);
}

// --- alpha * acc + beta * prior -------------------------------------------

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

// ===========================================================================
// The 16-byte access granule
// ===========================================================================
//
// Every vectorized load and store in this kernel is EXACTLY 16 bytes, whatever
// the scalar width. That is what keeps an 8-lane LDS phase on exactly the 32
// banks, and it is why this is not simply Packet4<T>: for double a 4-element
// packet is 32 bytes and its 8-lane phase straddles 256 bytes -- a 2-way bank
// conflict -- while for complex<double> it is 64 bytes and 4-way.

// may_alias: these types are used to read and write objects of type D, which
// is a strict-aliasing violation without it. -O3 is entitled to reorder the
// shared-memory stores against the fragment loads if it believes they cannot
// alias, and the barrier alone does not stop that.
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

// A raw 16-byte move, for staging (a pure bit copy, so the scalar type is
// irrelevant). This has to be a native LLVM vector type rather than a struct:
// a `struct{T v[N];}` copy is split by SROA into element loads and stores and
// the 16-byte form is lost -- measured, in the PTX, as four ld.global.b64 +
// four st.shared.b64 where two ld.global.v2.b64 + two st.shared.v2.b64 were
// intended. `struct` is fine for the fragment *loads*, which are consumed
// element-wise and stay vectorized; it is only the whole-granule copy that
// needs this.
typedef double Raw16Base __attribute__((ext_vector_type(2)));
typedef Raw16Base Raw16 __attribute__((may_alias));

inline Raw16 raw16_load(const void* p) { return *reinterpret_cast<const Raw16*>(p); }
inline void raw16_store(void* p, Raw16 v) { *reinterpret_cast<Raw16*>(p) = v; }

// ===========================================================================
// Tile geometry
// ===========================================================================

struct Tile {
    static constexpr int M = 64;
    static constexpr int N = 64;
    static constexpr int K = 16;
    static constexpr int TT = 4;            // thread tile, both directions
    static constexpr int LocalRows = M / TT;  // 16, the m direction
    static constexpr int LocalCols = N / TT;  // 16, the n direction
    static constexpr int Threads = LocalRows * LocalCols;  // 256
    // No padding. An aligned stride is what lets the fragment loads vectorize.
    static constexpr int AStride = M;
    static constexpr int BStride = N;
    static constexpr int PerThreadA = M * K / Threads;  // 4
    static constexpr int PerThreadB = K * N / Threads;  // 4
};

static_assert(Tile::M * Tile::K == Tile::PerThreadA * Tile::Threads, "A staging");
static_assert(Tile::K * Tile::N == Tile::PerThreadB * Tile::Threads, "B staging");

inline const char* tile_tag() { return "64x64x16_t4x4"; }

// Does the problem satisfy everything the unpredicated path assumes?
template <typename D>
bool fast_path_ok(int m, int n, int k,
                  const D* a, int lda, long long stride_a,
                  const D* b, int ldb, long long stride_b,
                  const D* c, int ldc, long long stride_c) {
    constexpr int VecN = Vec16<D>::N;
    if ((m % Tile::M) != 0 || (n % Tile::N) != 0 || (k % Tile::K) != 0) return false;
    auto ok = [](const D* p, int ld, long long st) {
        return p != nullptr &&
               (reinterpret_cast<std::uintptr_t>(p) % 16u) == 0 &&
               (ld % VecN) == 0 && (st % VecN) == 0;
    };
    return ok(a, lda, stride_a) && ok(b, ldb, stride_b) && ok(c, ldc, stride_c);
}

// ===========================================================================
// The kernel
// ===========================================================================

template <typename D, bool Fast>
class WideScalarGemmKernel;

template <typename D, bool Fast>
sycl::event launch_wsg(sycl::queue& q,
                       int m, int n, int k, int batch,
                       const D* a_ptr, int lda, long long stride_a,
                       const D* b_ptr, int ldb, long long stride_b,
                       D* c_ptr, int ldc, long long stride_c,
                       D alpha, D beta) {
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

    const int gm_tiles = (m + TileM - 1) / TileM;
    const int gn_tiles = (n + TileN - 1) / TileN;

    // local_id(2) is the fastest-varying SYCL dimension; it carries m.
    const sycl::range<3> local(1, LocalCols, LocalRows);
    const sycl::range<3> global(static_cast<size_t>(batch),
                                static_cast<size_t>(gn_tiles) * LocalCols,
                                static_cast<size_t>(gm_tiles) * LocalRows);

    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<Vec16<D>, 1> tile_a(sycl::range<1>(TileK * AStride / VecN), h);
        sycl::local_accessor<Vec16<D>, 1> tile_b(sycl::range<1>(TileK * BStride / VecN), h);

        h.parallel_for<WideScalarGemmKernel<D, Fast>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch) return;

                // m is the fastest-varying thread index. C is column-major, so
                // lanes that differ in m touch consecutive addresses; lanes
                // that differ in n would stride by ldc. Getting this backwards
                // is nearly free at beta == 0 and catastrophic at beta != 0,
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

                // Staging coordinates.
                //
                //  A is column-major m x k, contiguous down m. The tile is cut
                //  into 16-byte granules and granule ids are handed out in
                //  linear tid order, one chunk at a time. Consecutive lanes
                //  therefore get consecutive granules: the global read of a
                //  warp is 32 x 16 = 512 contiguous bytes, and the shared write
                //  of an 8-lane phase is 128 contiguous bytes = exactly the 32
                //  banks, conflict-free, for every scalar width.
                //
                //  (Handing each thread PerThreadA *consecutive elements* --
                //  what the float kernel does, where 4 floats happen to BE one
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
                //  PerThreadB consecutive k from one n-column and scatters them
                //  into shared so that shared ends up [k][n]. The scatter is
                //  irreducibly non-vector on the store side (that is the
                //  transpose), and it costs a 4-way shared-bank conflict; it is
                //  4 store instructions per k0 tile against 64 fragment loads,
                //  so it is left alone in favour of keeping the global read of
                //  B coalesced (4 lanes per column, 8 columns per warp).
                const int b_k = (tid % (TileK / Tile::PerThreadB)) * Tile::PerThreadB;  // 0,4,8,12
                const int b_n = tid / (TileK / Tile::PerThreadB);                       // 0..63

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    if constexpr (Fast) {
                        // ---- A: 16-byte global load -> 16-byte shared store --
#pragma unroll
                        for (int c = 0; c < Chunks; ++c) {
                            raw16_store(
                                &sa[a_gk[c] * AStride + a_gm[c]],
                                raw16_load(Ab + (m0 + a_gm[c]) +
                                           static_cast<std::ptrdiff_t>(k0 + a_gk[c]) * lda));
                        }
                        // ---- B: vectorized load, scattered shared store ----
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
                            for (int w = 0; w < Wb; ++w) af[band * Wb + w] = t.v[w];
                        }
#pragma unroll
                        for (int band = 0; band < NB; ++band) {
                            const Vec16<D> t =
                                vec_ref(&sb[kk * BStride + band * NSep + tx * Wb]);
#pragma unroll
                            for (int w = 0; w < Wb; ++w) bf[band * Wb + w] = t.v[w];
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

                // ---- Epilogue -------------------------------------------
                // Within a band the Wb rows are consecutive in m, which is the
                // contiguous direction of a column-major C, so a whole band is
                // one 16-byte access. Lanes differ in ty, i.e. in m, so the
                // band bases are consecutive too.
                const bool beta_zero = dev_is_zero(beta);
#pragma unroll
                for (int bm = 0; bm < NB; ++bm) {
                    const int gm = m0 + bm * MSep + ty * Wb;
#pragma unroll
                    for (int j = 0; j < TT; ++j) {
                        const int bn = j / Wb;
                        const int w_n = j % Wb;
                        const int gn = n0 + bn * NSep + tx * Wb + w_n;
                        if constexpr (Fast) {
                            D* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                            Vec16<D> out;
                            if (beta_zero) {
#pragma unroll
                                for (int w = 0; w < Wb; ++w) {
                                    out.v[w] = lin_epi(alpha, D{}, accum[bm * Wb + w][j], D{});
                                }
                            } else {
                                const Vec16<D> prior = vec_ref(const_cast<const D*>(p));
#pragma unroll
                                for (int w = 0; w < Wb; ++w) {
                                    out.v[w] = lin_epi(alpha, beta, accum[bm * Wb + w][j],
                                                       prior.v[w]);
                                }
                            }
                            vec_ref(p) = out;
                        } else {
                            if (gn >= n) continue;
#pragma unroll
                            for (int w = 0; w < Wb; ++w) {
                                const int row = gm + w;
                                if (row >= m) continue;
                                D* p = &Cb[row + static_cast<std::ptrdiff_t>(gn) * ldc];
                                const D prior = beta_zero ? D{} : *p;
                                *p = lin_epi(alpha, beta_zero ? D{} : beta,
                                             accum[bm * Wb + w][j], prior);
                            }
                        }
                    }
                }
            });
    });
}

// Dispatch on whether the unpredicated path is legal for this problem.
template <typename D>
sycl::event run_wsg(sycl::queue& q, bool& used_fast,
                    int m, int n, int k, int batch,
                    const D* a, int lda, long long sa,
                    const D* b, int ldb, long long sb,
                    D* c, int ldc, long long sc,
                    D alpha, D beta) {
    if (fast_path_ok<D>(m, n, k, a, lda, sa, b, ldb, sb, c, ldc, sc)) {
        used_fast = true;
        return launch_wsg<D, true>(q, m, n, k, batch, a, lda, sa, b, ldb, sb, c, ldc, sc,
                                   alpha, beta);
    }
    used_fast = false;
    return launch_wsg<D, false>(q, m, n, k, batch, a, lda, sa, b, ldb, sb, c, ldc, sc,
                                alpha, beta);
}

// ===========================================================================
// Host side
// ===========================================================================

template <typename T>
struct HostAcc {
    using type = double;
};
template <typename R>
struct HostAcc<std::complex<R>> {
    using type = std::complex<double>;
};

inline double to_acc(float x) { return double(x); }
inline double to_acc(double x) { return x; }
inline std::complex<double> to_acc(std::complex<float> x) {
    return std::complex<double>(double(x.real()), double(x.imag()));
}
inline std::complex<double> to_acc(std::complex<double> x) { return x; }

template <typename T>
struct MakeVal {
    static T val(double re, double) { return T(re); }
};
template <typename R>
struct MakeVal<std::complex<R>> {
    static std::complex<R> val(double re, double im) {
        return std::complex<R>(R(re), R(im));
    }
};

inline double u01(std::uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return double((s >> 33) & 0xFFFFFF) / double(0x1000000) - 0.5;
}

template <typename T>
void fill(std::vector<T>& v, std::uint64_t seed) {
    std::uint64_t s = seed;
    for (auto& x : v) {
        const double a = u01(s);
        const double b = u01(s);
        x = MakeVal<T>::val(a, b);
    }
}

// Small host reference: C = alpha*A*B + beta*C0, column-major, batched, NN.
template <typename T>
void host_ref(int m, int n, int k, int batch,
              const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C0,
              typename HostAcc<T>::type alpha, typename HostAcc<T>::type beta,
              std::vector<typename HostAcc<T>::type>& out) {
    using Acc = typename HostAcc<T>::type;
    out.assign(std::size_t(m) * n * batch, Acc(0));
    for (int b = 0; b < batch; ++b) {
        const std::size_t oa = std::size_t(b) * m * k;
        const std::size_t ob = std::size_t(b) * k * n;
        const std::size_t oc = std::size_t(b) * m * n;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                Acc acc(0);
                for (int p = 0; p < k; ++p) {
                    acc += to_acc(A[oa + std::size_t(p) * m + i]) *
                           to_acc(B[ob + std::size_t(j) * k + p]);
                }
                out[oc + std::size_t(j) * m + i] =
                    alpha * acc + beta * to_acc(C0[oc + std::size_t(j) * m + i]);
            }
        }
    }
}

// One correctness run at a given shape. Returns the normwise max relative
// error, measured against a double-precision host reference.
template <typename T>
double check_shape(sycl::queue& q, int m, int n, int k, int batch, double beta_real,
                   bool& used_fast) {
    using D = typename DevMap<T>::type;
    using Acc = typename HostAcc<T>::type;

    const std::size_t na = std::size_t(m) * k * batch;
    const std::size_t nb = std::size_t(k) * n * batch;
    const std::size_t nc = std::size_t(m) * n * batch;

    std::vector<T> hA(na), hB(nb), hC(nc);
    fill(hA, 0x9E3779B97F4A7C15ULL);
    fill(hB, 0xBF58476D1CE4E5B9ULL);
    fill(hC, 0x94D049BB133111EBULL);

    T* dA = sycl::malloc_device<T>(na, q);
    T* dB = sycl::malloc_device<T>(nb, q);
    T* dC = sycl::malloc_device<T>(nc, q);
    q.memcpy(dA, hA.data(), na * sizeof(T)).wait();
    q.memcpy(dB, hB.data(), nb * sizeof(T)).wait();
    q.memcpy(dC, hC.data(), nc * sizeof(T)).wait();

    const T alpha_h = MakeVal<T>::val(1.0, 0.0);
    const T beta_h = MakeVal<T>::val(beta_real, 0.0);
    D alpha{}, beta{};
    std::memcpy(&alpha, &alpha_h, sizeof(D));
    std::memcpy(&beta, &beta_h, sizeof(D));

    run_wsg<D>(q, used_fast, m, n, k, batch,
               reinterpret_cast<const D*>(dA), m, (long long)m * k,
               reinterpret_cast<const D*>(dB), k, (long long)k * n,
               reinterpret_cast<D*>(dC), m, (long long)m * n,
               alpha, beta)
        .wait();

    std::vector<T> got(nc);
    q.memcpy(got.data(), dC, nc * sizeof(T)).wait();

    std::vector<Acc> ref;
    host_ref<T>(m, n, k, batch, hA, hB, hC, Acc(1.0), Acc(beta_real), ref);

    double max_err = 0.0, max_ref = 0.0;
    for (std::size_t i = 0; i < nc; ++i) {
        const Acc d = to_acc(got[i]) - ref[i];
        max_err = std::max(max_err, double(std::abs(d)));
        max_ref = std::max(max_ref, double(std::abs(ref[i])));
    }

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return max_err / std::max(max_ref, 1e-300);
}

struct Opts {
    int m = 512, n = 512, k = 512, batch = 64;
    int iters = 30, warmup = 10;
    double beta = 1.0;
    std::string dtype = "double";
    bool skip_check = false;
    bool check_only = false;
};

template <typename T>
int run(const Opts& o) {
    using D = typename DevMap<T>::type;
    constexpr bool cplx = DevMap<T>::is_complex;

    sycl::queue q{sycl::default_selector_v};
    std::fprintf(stderr, "device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---- correctness first. A fast wrong kernel is worthless. -----------
    double maxrelerr = -1.0;
    if (!o.skip_check) {
        bool f1 = false, f2 = false;
        // Shape 1 exercises the unpredicated fast path (64 | m,n and 16 | k).
        const double e1 = check_shape<T>(q, 128, 128, 32, 3, o.beta, f1);
        // Shape 2 is ragged in all three dimensions -> predicated path.
        const double e2 = check_shape<T>(q, 70, 53, 37, 2, o.beta, f2);
        maxrelerr = std::max(e1, e2);
        std::fprintf(stderr,
                     "check: 128x128x32 b3 (%s) relerr=%.3e | 70x53x37 b2 (%s) relerr=%.3e\n",
                     f1 ? "fast" : "pred", e1, f2 ? "fast" : "pred", e2);
        const double tol = (sizeof(typename DevMap<T>::real) == 4) ? 5e-5 : 1e-12;
        if (!(maxrelerr <= tol)) {
            std::fprintf(stderr, "FAIL: relerr %.3e exceeds tolerance %.3e\n", maxrelerr, tol);
            return 2;
        }
    }
    if (o.check_only) {
        std::printf("RESULT dtype=%s m=0 n=0 k=0 batch=0 beta=%g tile=%s ms=0 tflops=0 "
                    "maxrelerr=%.3e\n",
                    o.dtype.c_str(), o.beta, tile_tag(), maxrelerr);
        return 0;
    }

    // ---- timed run -------------------------------------------------------
    const int m = o.m, n = o.n, k = o.k, batch = o.batch;
    const std::size_t na = std::size_t(m) * k * batch;
    const std::size_t nb = std::size_t(k) * n * batch;
    const std::size_t nc = std::size_t(m) * n * batch;

    std::vector<T> hA(na), hB(nb), hC(nc);
    fill(hA, 0x9E3779B97F4A7C15ULL);
    fill(hB, 0xBF58476D1CE4E5B9ULL);
    fill(hC, 0x94D049BB133111EBULL);

    T* dA = sycl::malloc_device<T>(na, q);
    T* dB = sycl::malloc_device<T>(nb, q);
    T* dC = sycl::malloc_device<T>(nc, q);
    if (!dA || !dB || !dC) {
        std::fprintf(stderr, "allocation failed\n");
        return 3;
    }
    q.memcpy(dA, hA.data(), na * sizeof(T)).wait();
    q.memcpy(dB, hB.data(), nb * sizeof(T)).wait();
    // C is initialized because beta defaults to 1 and the kernel READS it.
    q.memcpy(dC, hC.data(), nc * sizeof(T)).wait();

    const T alpha_h = MakeVal<T>::val(1.0, 0.0);
    const T beta_h = MakeVal<T>::val(o.beta, 0.0);
    D alpha{}, beta{};
    std::memcpy(&alpha, &alpha_h, sizeof(D));
    std::memcpy(&beta, &beta_h, sizeof(D));

    bool used_fast = false;
    auto launch = [&]() {
        return run_wsg<D>(q, used_fast, m, n, k, batch,
                          reinterpret_cast<const D*>(dA), m, (long long)m * k,
                          reinterpret_cast<const D*>(dB), k, (long long)k * n,
                          reinterpret_cast<D*>(dC), m, (long long)m * n,
                          alpha, beta);
    };

    for (int i = 0; i < o.warmup; ++i) launch();
    q.wait();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < o.iters; ++i) launch();
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / o.iters;
    const double flop = (cplx ? 8.0 : 2.0) * double(m) * double(n) * double(k) * double(batch);
    const double tflops = flop / (ms * 1e-3) / 1e12;

    char tag[64];
    std::snprintf(tag, sizeof(tag), "%s/%s", tile_tag(), used_fast ? "fast" : "pred");

    std::printf("RESULT dtype=%s m=%d n=%d k=%d batch=%d beta=%g tile=%s ms=%.4f "
                "tflops=%.3f maxrelerr=%.3e\n",
                o.dtype.c_str(), m, n, k, batch, o.beta, tag, ms, tflops, maxrelerr);

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return 0;
}

}  // namespace wsg

int main(int argc, char** argv) {
    wsg::Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_i = [&]() { return std::atoi(argv[++i]); };
        auto next_d = [&]() { return std::atof(argv[++i]); };
        if (a == "--m" && i + 1 < argc) o.m = next_i();
        else if (a == "--n" && i + 1 < argc) o.n = next_i();
        else if (a == "--k" && i + 1 < argc) o.k = next_i();
        else if (a == "--batch" && i + 1 < argc) o.batch = next_i();
        else if (a == "--iters" && i + 1 < argc) o.iters = next_i();
        else if (a == "--warmup" && i + 1 < argc) o.warmup = next_i();
        else if (a == "--beta" && i + 1 < argc) o.beta = next_d();
        else if (a == "--dtype" && i + 1 < argc) o.dtype = argv[++i];
        else if (a == "--skip-check") o.skip_check = true;
        else if (a == "--check-only") o.check_only = true;
        else {
            std::fprintf(stderr,
                         "usage: %s [--dtype float|double|cfloat|cdouble] [--m N] [--n N] "
                         "[--k N] [--batch N] [--beta X] [--iters N] [--warmup N] "
                         "[--skip-check] [--check-only]\n",
                         argv[0]);
            return 1;
        }
    }

    if (o.dtype == "double" || o.dtype == "f64") return wsg::run<double>(o);
    if (o.dtype == "float" || o.dtype == "f32") return wsg::run<float>(o);
    if (o.dtype == "cfloat" || o.dtype == "complex64" || o.dtype == "c64")
        return wsg::run<std::complex<float>>(o);
    if (o.dtype == "cdouble" || o.dtype == "complex128" || o.dtype == "c128")
        return wsg::run<std::complex<double>>(o);

    std::fprintf(stderr, "unknown dtype '%s'\n", o.dtype.c_str());
    return 1;
}

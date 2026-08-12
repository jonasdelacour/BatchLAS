// =====================================================================
// tile-128x64-k8-t8x4 : a register-tiled batched GEMM for WIDE SCALARS
// =====================================================================
//
// The 128x128x8 / 8x8 float kernel (src/sycl/gemm/register_128x128.hh) cannot
// be widened: 64 accumulators is 64 registers for float, but 128 for double or
// complex<float> and 256 for complex<double>, against a 255-register hardware
// limit. It spills.
//
// This kernel keeps every structural property that made that kernel fast and
// pays for the wider scalar in ONE place only: the macro tile is narrowed in n
// from 128 to 64, which halves the thread tile from 8x8 to 8x4 and so halves
// the accumulator count from 64 elements to 32.
//
//   macro tile   128 (m) x 64 (n) x 8 (k)
//   threads      256, arranged 16 (m) x 16 (n)
//   thread tile  8 (m) x 4 (n) = 32 accumulators
//
//   accumulator registers per thread
//       float             32
//       double            64
//       complex<float>    64
//       complex<double>  128
//
//   shared memory per work-group = TileK*(TileM+TileN)*sizeof(T)
//                                = 8*192*sizeof(T) = 1536 elements
//       float     6 KB      double          12 KB
//       cfloat   12 KB      cdouble         24 KB
//   (the 128x128x8 kernel needs 2048 elements: 16 KB for double, 32 for
//    cdouble, which on top of the register pressure is a second occupancy
//    cliff. 12 KB fits 8 work-groups in the 100 KB sm_89 budget.)
//
// The four load-bearing properties of the float kernel, preserved:
//
//  1. FFMA-to-shared-load ratio. Per k-step a thread does 8*4 = 32 scalar
//     MACs and issues (8/VecLen) + (4/VecLen) vector shared loads. Crucially
//     VecLen is defined in BYTES, not elements: VecLen = 16/sizeof(T), so
//     EVERY fragment load is exactly one 16-byte LDS.128 whatever the scalar
//     is. That gives, per k-step:
//         float     3 loads /  32 FFMA          = 10.7 : 1
//         double    6 loads /  32 DFMA          =  5.3 : 1
//         cfloat    6 loads / 128 FFMA          = 21.3 : 1   (4 FFMA per MAC)
//         cdouble  12 loads / 128 DFMA          = 10.7 : 1
//     complex<float> beats the float kernel's 16:1 outright, because a complex
//     MAC is four FFMAs against one shared element pair. For double the
//     absolute ratio is lower, but DFMA on a consumer sm_89 runs at 1/64 the
//     FFMA rate, so the FP pipe is 64x further from saturation and the shared
//     pipe cannot be the limit.
//
//     This is why the shape narrows in n and not in m: narrowing n costs
//     B-fragment loads (4 -> 2 per k-step scaled), which is the cheaper half.
//
//  2. Aligned shared strides. AStride == TileM and BStride == TileN exactly,
//     never TileM+1. Both are multiples of VecLen for every scalar width, so
//     every fragment address is a multiple of 16 bytes and the compiler can
//     prove it. An odd stride degrades every fragment load to scalar
//     ld.shared.b32. The tiles are also allocated as a local_accessor of a
//     16-byte-aligned Packet type so the BASE is provably aligned too.
//
//  3. B staged as [k][n], not [n][k], so a thread's n values are contiguous
//     and vectorize.
//
//  4. Bank-conflict-free vectorized loads via the band split, generalized.
//     The float kernel splits the thread's 8 rows into two 4-wide bands 64
//     apart. Here the thread's ThreadTile elements split into ThreadTile/
//     VecLen bands, each band a VecLen-wide aligned quad, with bands spaced
//     (Local*VecLen) apart -- which always tiles the macro tile exactly, since
//     Local*VecLen*(ThreadTile/VecLen) = Local*ThreadTile = Tile. A 16-byte
//     LDS is serviced 8 lanes per phase and 8 lanes x 16 B = 128 B = exactly
//     the 32 banks, so consecutive-ty lanes reading consecutive quads are
//     conflict free at every scalar width.
//
// Two further things this kernel must get right, both previously measured:
//
//   * THE EPILOGUE. The m index is the FASTEST-VARYING thread index
//     (local_id(2)), so lanes that differ in m touch consecutive addresses in
//     a column-major C. Getting this backwards is nearly free at beta == 0 and
//     catastrophic at beta != 0, where the read of C becomes one scattered
//     transaction per lane. This harness therefore DEFAULTS TO beta = 1.
//
//   * THE COMPLEX MULTIPLY. std::complex operator* emits an isnan branch and a
//     call to __mulsc3 in device code. Nothing here uses it: complex is
//     re-typed to a POD Cplx and every product is written out as
//     (a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re) as four explicit fmas.
//
//   * Accumulators are a plain local array, and every arithmetic helper takes
//     and RETURNS BY VALUE. An out-parameter passed by reference has been
//     measured to spill the accumulator array in this codebase and cost 43%.
//
// Build:
//   /opt/dpcpp-cuda/bin/clang++ -O3 -std=c++20 -fsycl \
//     -fsycl-targets=nvidia_gpu_sm_89 --cuda-path=/usr/local/cuda-13.2 \
//     -Xcuda-ptxas -v tile-128x64-k8-t8x4.cpp -o tile-128x64-k8-t8x4
//
// Run:
//   ./tile-128x64-k8-t8x4 --dtype double --m 512 --n 512 --k 512 --batch 256
//   ./tile-128x64-k8-t8x4 --dtype cfloat --m 512 --n 512 --k 512 --batch 256
//
// A reference check on a small aligned shape AND a small ragged shape (which
// exercises the predicated path) runs by default before the timed sweep.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------------
// Scalar layer
// ---------------------------------------------------------------------

// A POD complex. Layout-compatible with std::complex<R> (the standard
// guarantees std::complex<R> has the same layout as R[2]), so device pointers
// can simply be reinterpreted. Nothing in device code ever touches
// std::complex, which is the entire point: its operator* is Annex-G guarded.
template <typename R>
struct Cplx {
    R re;
    R im;
};

template <typename T>
struct DevScalar {
    using type = T;
    using real = T;
    static constexpr bool is_complex = false;
};

template <typename R>
struct DevScalar<std::complex<R>> {
    using type = Cplx<R>;
    using real = R;
    static constexpr bool is_complex = true;
};

template <typename T>
using dev_scalar_t = typename DevScalar<T>::type;

// --- construction / predicates ---------------------------------------

// Partial specialisation on a class template rather than function specialisation,
// so there is no primary-template branch naming members that only exist for one
// of the two cases.
template <typename W>
struct DevZero {
    static W get() { return W(0); }
};
template <typename R>
struct DevZero<Cplx<R>> {
    static Cplx<R> get() { return Cplx<R>{R(0), R(0)}; }
};

template <typename W>
inline W dev_zero() {
    return DevZero<W>::get();
}

template <typename R>
inline bool dev_is_zero(R x) {
    return x == R(0);
}
template <typename R>
inline bool dev_is_zero(Cplx<R> x) {
    return x.re == R(0) && x.im == R(0);
}

// --- the hot arithmetic ----------------------------------------------
//
// Every one of these takes its accumulator BY VALUE and RETURNS BY VALUE.
// No out-parameters: a reference into the accumulator array has been measured
// to force it out of registers.

template <typename R>
inline R dev_fma(R acc, R a, R b) {
    return sycl::fma(a, b, acc);
}

// The explicit complex MAC: exactly four fmas, no isnan branch, no __mulsc3.
//   re += a.re*b.re - a.im*b.im
//   im += a.re*b.im + a.im*b.re
template <typename R>
inline Cplx<R> dev_fma(Cplx<R> acc, Cplx<R> a, Cplx<R> b) {
    R re = sycl::fma(a.re, b.re, acc.re);
    re = sycl::fma(-a.im, b.im, re);
    R im = sycl::fma(a.re, b.im, acc.im);
    im = sycl::fma(a.im, b.re, im);
    return Cplx<R>{re, im};
}

// Epilogue: alpha*acc + beta*prior, again written out for complex.
template <typename R>
inline R dev_epilogue(R alpha, R acc, R beta, R prior) {
    return sycl::fma(alpha, acc, beta * prior);
}

template <typename R>
inline Cplx<R> dev_epilogue(Cplx<R> alpha, Cplx<R> acc, Cplx<R> beta, Cplx<R> prior) {
    R re = sycl::fma(beta.re, prior.re, R(0));
    re = sycl::fma(-beta.im, prior.im, re);
    R im = sycl::fma(beta.re, prior.im, R(0));
    im = sycl::fma(beta.im, prior.re, im);
    re = sycl::fma(alpha.re, acc.re, re);
    re = sycl::fma(-alpha.im, acc.im, re);
    im = sycl::fma(alpha.re, acc.im, im);
    im = sycl::fma(alpha.im, acc.re, im);
    return Cplx<R>{re, im};
}

// alpha*acc alone, for the beta == 0 branch.
template <typename R>
inline R dev_scale(R alpha, R acc) {
    return alpha * acc;
}
template <typename R>
inline Cplx<R> dev_scale(Cplx<R> alpha, Cplx<R> acc) {
    R re = sycl::fma(alpha.re, acc.re, R(0));
    re = sycl::fma(-alpha.im, acc.im, re);
    R im = sycl::fma(alpha.re, acc.im, R(0));
    im = sycl::fma(alpha.im, acc.re, im);
    return Cplx<R>{re, im};
}

// ---------------------------------------------------------------------
// The packet: ALWAYS 16 bytes
// ---------------------------------------------------------------------
//
// This is the single most important generalisation over Packet4<T>, which was
// alignas(4*sizeof(T)) and so demanded a 32-byte access for double and a
// 64-byte one for complex<double> -- neither of which exists as one SASS
// instruction. Here the ELEMENT COUNT shrinks with the scalar width so the
// BYTE WIDTH stays at 16, which is exactly LDS.128 / STG.E.128.
//
//   float 4 elems | double 2 | complex<float> 2 | complex<double> 1
//
// complex<double> is 16 bytes on its own, so VecLen == 1 there is still a
// full-width 128-bit access, not a scalar fallback.

template <typename W>
constexpr int vec_len() {
    return (sizeof(W) >= 16) ? 1 : static_cast<int>(16 / sizeof(W));
}

template <typename W, int VL>
struct alignas(VL * sizeof(W)) Packet {
    W v[VL];
};

template <typename W, int VL>
inline const Packet<W, VL>& packet_ref(const W* p) {
    return *reinterpret_cast<const Packet<W, VL>*>(p);
}

template <typename W, int VL>
inline Packet<W, VL>& packet_ref(W* p) {
    return *reinterpret_cast<Packet<W, VL>*>(p);
}

// ---------------------------------------------------------------------
// Tile geometry
// ---------------------------------------------------------------------

struct Tile128x64K8 {
    static constexpr int TileM = 128;
    static constexpr int TileN = 64;
    static constexpr int TileK = 8;
    static constexpr int ThreadTileM = 8;
    static constexpr int ThreadTileN = 4;
    static constexpr int LocalRows = TileM / ThreadTileM;  // 16
    static constexpr int LocalCols = TileN / ThreadTileN;  // 16
    static constexpr int Threads = LocalRows * LocalCols;  // 256
    // No padding. An aligned stride is what lets the fragment loads vectorize.
    static constexpr int AStride = TileM;
    static constexpr int BStride = TileN;
    static constexpr const char* name = "128x64x8_t8x4";
};

// Does this problem satisfy everything the unpredicated path assumes?
template <typename T, typename Geom>
bool can_use_fast_path(int m, int n, int k,
                       const T* a, int lda, std::int64_t stride_a,
                       const T* b, int ldb, std::int64_t stride_b,
                       const T* c, int ldc, std::int64_t stride_c) {
    using W = dev_scalar_t<T>;
    constexpr int VL = vec_len<W>();
    if ((m % Geom::TileM) != 0 || (n % Geom::TileN) != 0 || (k % Geom::TileK) != 0) {
        return false;
    }
    auto ok = [](const T* p, int ld, std::int64_t stride) {
        return p != nullptr &&
               (reinterpret_cast<std::uintptr_t>(p) % (VL * sizeof(W))) == 0 &&
               (ld % VL) == 0 && (stride % VL) == 0;
    };
    return ok(a, lda, stride_a) && ok(b, ldb, stride_b) && ok(c, ldc, stride_c);
}

template <typename T, bool Fast>
class WideGemmKernel;

// ---------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------

template <typename T, typename Geom, bool Fast>
sycl::event launch_wide_gemm(sycl::queue& q,
                             int m, int n, int k, int batch,
                             const T* a_in, int lda, std::int64_t stride_a,
                             const T* b_in, int ldb, std::int64_t stride_b,
                             T* c_in, int ldc, std::int64_t stride_c,
                             T alpha_in, T beta_in) {
    using W = dev_scalar_t<T>;
    static_assert(sizeof(W) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int TileM = Geom::TileM;
    constexpr int TileN = Geom::TileN;
    constexpr int TileK = Geom::TileK;
    constexpr int TTM = Geom::ThreadTileM;
    constexpr int TTN = Geom::ThreadTileN;
    constexpr int LocalRows = Geom::LocalRows;
    constexpr int LocalCols = Geom::LocalCols;
    constexpr int Threads = Geom::Threads;
    constexpr int AStride = Geom::AStride;
    constexpr int BStride = Geom::BStride;

    constexpr int VL = vec_len<W>();
    using Pk = Packet<W, VL>;

    // The band split, generalised. A thread's TTM rows are TTM/VL aligned
    // quads, spaced BandSpanM apart; and BandSpanM * NBandM == TileM exactly,
    // so the bands tile the macro tile with no remainder at any scalar width.
    constexpr int NBandM = TTM / VL;
    constexpr int NBandN = TTN / VL;
    constexpr int BandSpanM = LocalRows * VL;
    constexpr int BandSpanN = LocalCols * VL;
    static_assert(TTM % VL == 0 && TTN % VL == 0, "thread tile must divide by VL");
    static_assert(NBandM * BandSpanM == TileM, "m bands must tile TileM");
    static_assert(NBandN * BandSpanN == TileN, "n bands must tile TileN");
    static_assert(AStride % VL == 0 && BStride % VL == 0, "shared strides must be VL-aligned");

    // Staging geometry. A is read down m (contiguous in a column-major A) so
    // the warp is coalesced; B is read down k and transposed into shared so
    // that n ends up contiguous.
    constexpr int AThreadsPerK = TileM / VL;      // 32 / 64 / 128
    constexpr int AKStep = Threads / AThreadsPerK;  // 8 / 4 / 2
    constexpr int BThreadsPerN = TileK / VL;      // 2 / 4 / 8
    constexpr int BNStep = Threads / BThreadsPerN;  // 128 / 64 / 32
    static_assert(TileM % VL == 0 && TileK % VL == 0, "tiles must divide by VL");
    static_assert(AThreadsPerK <= Threads && BThreadsPerN <= Threads, "staging fits");
    static_assert(Threads % AThreadsPerK == 0 && Threads % BThreadsPerN == 0, "even staging");

    constexpr int SharedElems = TileK * (AStride + BStride);
    constexpr int SharedPackets = SharedElems / VL;
    static_assert(SharedElems % VL == 0, "shared tile must be a whole number of packets");

    const int group_rows = (m + TileM - 1) / TileM;
    const int group_cols = (n + TileN - 1) / TileN;

    const sycl::range<3> local(1, LocalRows, LocalCols);
    const sycl::range<3> global(static_cast<size_t>(batch),
                                static_cast<size_t>(group_rows) * LocalRows,
                                static_cast<size_t>(group_cols) * LocalCols);

    const W* a_ptr = reinterpret_cast<const W*>(a_in);
    const W* b_ptr = reinterpret_cast<const W*>(b_in);
    W* c_ptr = reinterpret_cast<W*>(c_in);
    W alpha, beta;
    std::memcpy(&alpha, &alpha_in, sizeof(W));
    std::memcpy(&beta, &beta_in, sizeof(W));

    return q.submit([&](sycl::handler& h) {
        // Allocated as packets so the BASE of the tile is provably 16-byte
        // aligned; a local_accessor<W,1> would only promise alignof(W).
        sycl::local_accessor<Pk, 1> tile(sycl::range<1>(SharedPackets), h);

        h.parallel_for<WideGemmKernel<T, Fast>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch) {
                    return;
                }

                // The m index is the FASTEST-VARYING one. C is column-major,
                // so lanes differing in m touch consecutive addresses, while
                // lanes differing in n would stride by ldc. Getting this
                // backwards is free at beta == 0 and catastrophic at beta != 0.
                const int ty = static_cast<int>(item.get_local_id(2));  // 0..15, m
                const int tx = static_cast<int>(item.get_local_id(1));  // 0..15, n
                const int tid = tx * LocalRows + ty;

                const int m0 = static_cast<int>(item.get_group(1)) * TileM;
                const int n0 = static_cast<int>(item.get_group(2)) * TileN;

                const W* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                const W* Bb = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
                W* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;

                W* sa = reinterpret_cast<W*>(
                    tile.template get_multi_ptr<sycl::access::decorated::no>().get());
                W* sb = sa + TileK * AStride;

                // 8x4 = 32 accumulators. A plain local array, nothing else.
                W accum[TTM][TTN];
#pragma unroll
                for (int i = 0; i < TTM; ++i) {
#pragma unroll
                    for (int j = 0; j < TTN; ++j) {
                        accum[i][j] = dev_zero<W>();
                    }
                }

                const int a_m = (tid % AThreadsPerK) * VL;
                const int a_k0 = tid / AThreadsPerK;
                const int b_k = (tid % BThreadsPerN) * VL;
                const int b_n0 = tid / BThreadsPerN;

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    if constexpr (Fast) {
                        // A: one 16-byte global load -> one 16-byte shared store.
#pragma unroll
                        for (int ak = a_k0; ak < TileK; ak += AKStep) {
                            packet_ref<W, VL>(&sa[ak * AStride + a_m]) =
                                packet_ref<W, VL>(Ab + (m0 + a_m) +
                                                  static_cast<std::ptrdiff_t>(k0 + ak) * lda);
                        }
                        // B: one 16-byte global load down k, then scattered
                        // into shared as [k][n]. The transpose cannot be
                        // vectorized on the store side; that is inherent.
#pragma unroll
                        for (int bn = b_n0; bn < TileN; bn += BNStep) {
                            const Pk vb = packet_ref<W, VL>(
                                Bb + (k0 + b_k) +
                                static_cast<std::ptrdiff_t>(n0 + bn) * ldb);
#pragma unroll
                            for (int i = 0; i < VL; ++i) {
                                sb[(b_k + i) * BStride + bn] = vb.v[i];
                            }
                        }
                    } else {
                        // Predicated staging. The shared tile is always filled
                        // to its full extent, with zeros outside the matrix, so
                        // the inner loop needs no bounds checks at all and is
                        // bit-identical between the two paths.
#pragma unroll
                        for (int ak = a_k0; ak < TileK; ak += AKStep) {
                            const int gk = k0 + ak;
#pragma unroll
                            for (int i = 0; i < VL; ++i) {
                                const int gm = m0 + a_m + i;
                                sa[ak * AStride + a_m + i] =
                                    (gm < m && gk < k)
                                        ? Ab[gm + static_cast<std::ptrdiff_t>(gk) * lda]
                                        : dev_zero<W>();
                            }
                        }
#pragma unroll
                        for (int bn = b_n0; bn < TileN; bn += BNStep) {
                            const int gn = n0 + bn;
#pragma unroll
                            for (int i = 0; i < VL; ++i) {
                                const int gk = k0 + b_k + i;
                                sb[(b_k + i) * BStride + bn] =
                                    (gk < k && gn < n)
                                        ? Bb[gk + static_cast<std::ptrdiff_t>(gn) * ldb]
                                        : dev_zero<W>();
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
                    for (int kk = 0; kk < TileK; ++kk) {
                        // NBandM + NBandN vectorized shared loads, each one a
                        // 16-byte LDS.128, feeding 32 MACs.
                        W af[TTM];
                        W bf[TTN];
#pragma unroll
                        for (int bm = 0; bm < NBandM; ++bm) {
                            const Pk p = packet_ref<W, VL>(
                                &sa[kk * AStride + bm * BandSpanM + ty * VL]);
#pragma unroll
                            for (int e = 0; e < VL; ++e) {
                                af[bm * VL + e] = p.v[e];
                            }
                        }
#pragma unroll
                        for (int bn = 0; bn < NBandN; ++bn) {
                            const Pk p = packet_ref<W, VL>(
                                &sb[kk * BStride + bn * BandSpanN + tx * VL]);
#pragma unroll
                            for (int e = 0; e < VL; ++e) {
                                bf[bn * VL + e] = p.v[e];
                            }
                        }
#pragma unroll
                        for (int i = 0; i < TTM; ++i) {
#pragma unroll
                            for (int j = 0; j < TTN; ++j) {
                                accum[i][j] = dev_fma(accum[i][j], af[i], bf[j]);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Epilogue. Within a band the VL rows are consecutive in m,
                // which is the contiguous direction of a column-major C, so a
                // whole band is one 16-byte store (and, at beta != 0, one
                // 16-byte read rather than VL scalar ones).
                const bool beta_zero = dev_is_zero(beta);
#pragma unroll
                for (int bm = 0; bm < NBandM; ++bm) {
                    const int gm = m0 + bm * BandSpanM + ty * VL;
#pragma unroll
                    for (int j = 0; j < TTN; ++j) {
                        const int gn = n0 + (j / VL) * BandSpanN + tx * VL + (j % VL);
                        if constexpr (Fast) {
                            W* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                            Pk out;
                            if (beta_zero) {
#pragma unroll
                                for (int e = 0; e < VL; ++e) {
                                    out.v[e] = dev_scale(alpha, accum[bm * VL + e][j]);
                                }
                            } else {
                                const Pk prior = packet_ref<W, VL>(const_cast<const W*>(p));
#pragma unroll
                                for (int e = 0; e < VL; ++e) {
                                    out.v[e] = dev_epilogue(alpha, accum[bm * VL + e][j],
                                                            beta, prior.v[e]);
                                }
                            }
                            packet_ref<W, VL>(p) = out;
                        } else {
                            if (gn >= n) {
                                continue;
                            }
#pragma unroll
                            for (int e = 0; e < VL; ++e) {
                                const int row = gm + e;
                                if (row >= m) {
                                    continue;
                                }
                                W* p = &Cb[row + static_cast<std::ptrdiff_t>(gn) * ldc];
                                *p = beta_zero
                                         ? dev_scale(alpha, accum[bm * VL + e][j])
                                         : dev_epilogue(alpha, accum[bm * VL + e][j], beta, *p);
                            }
                        }
                    }
                }
            });
    });
}

// Route to the unpredicated path when the problem allows it.
template <typename T, typename Geom>
sycl::event dispatch_wide_gemm(sycl::queue& q,
                               int m, int n, int k, int batch,
                               const T* a, int lda, std::int64_t stride_a,
                               const T* b, int ldb, std::int64_t stride_b,
                               T* c, int ldc, std::int64_t stride_c,
                               T alpha, T beta) {
    if (can_use_fast_path<T, Geom>(m, n, k, a, lda, stride_a, b, ldb, stride_b,
                                   c, ldc, stride_c)) {
        return launch_wide_gemm<T, Geom, true>(q, m, n, k, batch, a, lda, stride_a,
                                               b, ldb, stride_b, c, ldc, stride_c,
                                               alpha, beta);
    }
    return launch_wide_gemm<T, Geom, false>(q, m, n, k, batch, a, lda, stride_a,
                                            b, ldb, stride_b, c, ldc, stride_c,
                                            alpha, beta);
}

// ---------------------------------------------------------------------
// Host-side plumbing
// ---------------------------------------------------------------------

using cd = std::complex<double>;

template <typename T>
cd to_cd(const T& x) {
    if constexpr (DevScalar<T>::is_complex) {
        return cd(static_cast<double>(x.real()), static_cast<double>(x.imag()));
    } else {
        return cd(static_cast<double>(x), 0.0);
    }
}

template <typename T>
T from_cd(const cd& x) {
    if constexpr (DevScalar<T>::is_complex) {
        using R = typename DevScalar<T>::real;
        return T(static_cast<R>(x.real()), static_cast<R>(x.imag()));
    } else {
        return static_cast<T>(x.real());
    }
}

// Deterministic fill in [-0.5, 0.5).
template <typename T>
void fill(std::vector<T>& v, std::uint32_t a, std::uint32_t c) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        auto draw = [&](std::uint32_t salt) {
            std::uint32_t s = static_cast<std::uint32_t>(i) * a + c + salt;
            return static_cast<double>(s % 1000u) / 1000.0 - 0.5;
        };
        v[i] = from_cd<T>(cd(draw(0u), DevScalar<T>::is_complex ? draw(7919u) : 0.0));
    }
}

struct CheckResult {
    double maxrelerr;
    bool used_fast;
};

// Column-major host reference in double precision, plus a max-norm relative
// error. A fast wrong kernel is worthless, so this runs by default.
template <typename T, typename Geom>
CheckResult reference_check(sycl::queue& q, int m, int n, int k, int batch,
                            const T& alpha, const T& beta) {
    const std::size_t ea = static_cast<std::size_t>(m) * k * batch;
    const std::size_t eb = static_cast<std::size_t>(k) * n * batch;
    const std::size_t ec = static_cast<std::size_t>(m) * n * batch;

    std::vector<T> hA(ea), hB(eb), hC(ec), hOut(ec);
    fill(hA, 1103515245u, 12345u);
    fill(hB, 22695477u, 1u);
    fill(hC, 69069u, 5u);

    T* dA = sycl::malloc_device<T>(ea, q);
    T* dB = sycl::malloc_device<T>(eb, q);
    T* dC = sycl::malloc_device<T>(ec, q);
    q.memcpy(dA, hA.data(), ea * sizeof(T)).wait();
    q.memcpy(dB, hB.data(), eb * sizeof(T)).wait();
    q.memcpy(dC, hC.data(), ec * sizeof(T)).wait();

    const int lda = m, ldb = k, ldc = m;
    const std::int64_t sa = static_cast<std::int64_t>(m) * k;
    const std::int64_t sb = static_cast<std::int64_t>(k) * n;
    const std::int64_t sc = static_cast<std::int64_t>(m) * n;

    const bool fast = can_use_fast_path<T, Geom>(m, n, k, dA, lda, sa, dB, ldb, sb,
                                                 dC, ldc, sc);
    dispatch_wide_gemm<T, Geom>(q, m, n, k, batch, dA, lda, sa, dB, ldb, sb,
                                dC, ldc, sc, alpha, beta);
    q.wait();
    q.memcpy(hOut.data(), dC, ec * sizeof(T)).wait();

    const cd al = to_cd(alpha), be = to_cd(beta);
    double maxdiff = 0.0, maxref = 0.0;
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                cd acc(0.0, 0.0);
                for (int p = 0; p < k; ++p) {
                    acc += to_cd(hA[static_cast<std::size_t>(b) * sa + p * lda + i]) *
                           to_cd(hB[static_cast<std::size_t>(b) * sb + j * ldb + p]);
                }
                const std::size_t idx = static_cast<std::size_t>(b) * sc + j * ldc + i;
                const cd want = al * acc + be * to_cd(hC[idx]);
                const cd got = to_cd(hOut[idx]);
                maxdiff = std::max(maxdiff, std::abs(got - want));
                maxref = std::max(maxref, std::abs(want));
            }
        }
    }

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return CheckResult{maxdiff / std::max(maxref, 1e-300), fast};
}

template <typename T>
double flops_per_mac() {
    return DevScalar<T>::is_complex ? 8.0 : 2.0;
}

template <typename T, typename Geom>
int run(int m, int n, int k, int batch, double beta_val, int iters, int warmup,
        const char* dtype_name, bool skip_timed) {
    sycl::queue q{sycl::gpu_selector_v};
    std::fprintf(stderr, "# device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());
    std::fprintf(stderr, "# tile %s  vec_len=%d elems (%zu bytes)  smem/wg=%zu bytes\n",
                 Geom::name, vec_len<dev_scalar_t<T>>(),
                 vec_len<dev_scalar_t<T>>() * sizeof(T),
                 static_cast<std::size_t>(Geom::TileK) *
                     (Geom::AStride + Geom::BStride) * sizeof(T));

    const T alpha = from_cd<T>(cd(1.0, 0.0));
    const T beta = from_cd<T>(cd(beta_val, 0.0));

    // Correctness first, on a small aligned shape (fast path) and a small
    // ragged one (predicated path). Both at the requested beta.
    const CheckResult c1 = reference_check<T, Geom>(q, Geom::TileM, Geom::TileN,
                                                    Geom::TileK * 3, 3, alpha, beta);
    const CheckResult c2 = reference_check<T, Geom>(q, 67, 45, 23, 2, alpha, beta);
    const double maxrelerr = std::max(c1.maxrelerr, c2.maxrelerr);
    const double tol = (sizeof(typename DevScalar<T>::real) == 4) ? 1e-5 : 1e-12;
    std::fprintf(stderr,
                 "# check aligned(fast=%d) %.3e   ragged(fast=%d) %.3e   tol %.1e -> %s\n",
                 static_cast<int>(c1.used_fast), c1.maxrelerr,
                 static_cast<int>(c2.used_fast), c2.maxrelerr, tol,
                 maxrelerr <= tol ? "PASS" : "FAIL");
    if (maxrelerr > tol) {
        std::fprintf(stderr, "# ABORT: reference check failed, timing suppressed\n");
        std::printf("RESULT dtype=%s m=%d n=%d k=%d batch=%d beta=%g tile=%s "
                    "ms=nan tflops=nan maxrelerr=%.6e\n",
                    dtype_name, m, n, k, batch, beta_val, Geom::name, maxrelerr);
        return 1;
    }
    if (skip_timed) {
        std::printf("RESULT dtype=%s m=%d n=%d k=%d batch=%d beta=%g tile=%s "
                    "ms=nan tflops=nan maxrelerr=%.6e\n",
                    dtype_name, m, n, k, batch, beta_val, Geom::name, maxrelerr);
        return 0;
    }

    const std::size_t ea = static_cast<std::size_t>(m) * k * batch;
    const std::size_t eb = static_cast<std::size_t>(k) * n * batch;
    const std::size_t ec = static_cast<std::size_t>(m) * n * batch;

    T* dA = sycl::malloc_device<T>(ea, q);
    T* dB = sycl::malloc_device<T>(eb, q);
    T* dC = sycl::malloc_device<T>(ec, q);
    if (!dA || !dB || !dC) {
        std::fprintf(stderr, "# allocation failed\n");
        return 2;
    }
    {
        std::vector<T> hA(ea), hB(eb), hC(ec);
        fill(hA, 1103515245u, 12345u);
        fill(hB, 22695477u, 1u);
        fill(hC, 69069u, 5u);
        q.memcpy(dA, hA.data(), ea * sizeof(T)).wait();
        q.memcpy(dB, hB.data(), eb * sizeof(T)).wait();
        // C must be initialised: at beta != 0 the kernel reads it, and reading
        // uninitialised device memory can produce NaN and poison the timing.
        q.memcpy(dC, hC.data(), ec * sizeof(T)).wait();
    }

    const int lda = m, ldb = k, ldc = m;
    const std::int64_t sa = static_cast<std::int64_t>(m) * k;
    const std::int64_t sb = static_cast<std::int64_t>(k) * n;
    const std::int64_t sc = static_cast<std::int64_t>(m) * n;

    const bool fast = can_use_fast_path<T, Geom>(m, n, k, dA, lda, sa, dB, ldb, sb,
                                                 dC, ldc, sc);
    std::fprintf(stderr, "# timed shape takes the %s path\n",
                 fast ? "unpredicated fast" : "predicated");

    auto launch = [&]() {
        return dispatch_wide_gemm<T, Geom>(q, m, n, k, batch, dA, lda, sa, dB, ldb, sb,
                                           dC, ldc, sc, alpha, beta);
    };

    for (int i = 0; i < warmup; ++i) launch();
    q.wait();

    // Wall clock only. Summing per-event command_start..command_end over
    // queued submissions has been measured to inflate by 6x here.
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) launch();
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double flops = flops_per_mac<T>() * static_cast<double>(m) * n * k * batch;
    const double tflops = flops / (ms * 1e-3) / 1e12;

    std::printf("RESULT dtype=%s m=%d n=%d k=%d batch=%d beta=%g tile=%s "
                "ms=%.4f tflops=%.3f maxrelerr=%.6e\n",
                dtype_name, m, n, k, batch, beta_val, Geom::name, ms, tflops, maxrelerr);

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return 0;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s [--m M] [--n N] [--k K] [--batch B] [--dtype "
                 "float|double|cfloat|cdouble] [--beta X] [--iters I] [--warmup W] "
                 "[--check-only]\n"
                 "   or: %s M N K BATCH DTYPE BETA\n",
                 prog, prog);
}

int main(int argc, char** argv) {
    int m = 512, n = 512, k = 512, batch = 128;
    int iters = 20, warmup = 5;
    double beta_val = 1.0;  // NOT 0: a harness defaulting to beta=0 cannot see
                            // the scattered-epilogue defect.
    std::string dtype = "double";
    bool check_only = false;

    if (argc > 1 && argv[1][0] != '-') {
        // Positional: m n k batch dtype beta
        const int np = argc - 1;
        if (np >= 1) m = std::atoi(argv[1]);
        if (np >= 2) n = std::atoi(argv[2]);
        if (np >= 3) k = std::atoi(argv[3]);
        if (np >= 4) batch = std::atoi(argv[4]);
        if (np >= 5) dtype = argv[5];
        if (np >= 6) beta_val = std::atof(argv[6]);
    } else {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto next_i = [&]() { return (i + 1 < argc) ? std::atoi(argv[++i]) : 0; };
            auto next_d = [&]() { return (i + 1 < argc) ? std::atof(argv[++i]) : 0.0; };
            auto next_s = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
            if (a == "--m") m = next_i();
            else if (a == "--n") n = next_i();
            else if (a == "--k") k = next_i();
            else if (a == "--batch") batch = next_i();
            else if (a == "--iters") iters = next_i();
            else if (a == "--warmup") warmup = next_i();
            else if (a == "--beta") beta_val = next_d();
            else if (a == "--dtype") dtype = next_s();
            else if (a == "--check-only") check_only = true;
            else { usage(argv[0]); return 2; }
        }
    }

    if (m <= 0 || n <= 0 || k <= 0 || batch <= 0) {
        usage(argv[0]);
        return 2;
    }

    using G = Tile128x64K8;
    if (dtype == "double") {
        return run<double, G>(m, n, k, batch, beta_val, iters, warmup, "double", check_only);
    } else if (dtype == "float") {
        return run<float, G>(m, n, k, batch, beta_val, iters, warmup, "float", check_only);
    } else if (dtype == "cfloat" || dtype == "complex64") {
        return run<std::complex<float>, G>(m, n, k, batch, beta_val, iters, warmup,
                                           "cfloat", check_only);
    } else if (dtype == "cdouble" || dtype == "complex128") {
        return run<std::complex<double>, G>(m, n, k, batch, beta_val, iters, warmup,
                                            "cdouble", check_only);
    }
    std::fprintf(stderr, "unknown dtype '%s'\n", dtype.c_str());
    usage(argv[0]);
    return 2;
}

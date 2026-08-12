// tile-complex-split.cpp
//
// A batched register-tiled SYCL GEMM for WIDE SCALARS (double, complex<float>,
// complex<double>), modelled on src/sycl/gemm/register_128x128.hh but with a
// shared-memory layout and a tile shape chosen for the wide scalar rather than
// inherited from the float kernel.
//
// ===========================================================================
// 1. THE HEADLINE: A COMPLEX MAC IS TWICE AS FMA-RICH PER LOADED WORD
// ===========================================================================
// The float kernel's whole argument is the FMA : shared-load ratio.  For a
// TM x TN thread tile, per k-step:
//
//     128-bit shared loads = (TM + TN) * planes / W    [W = 16/sizeof(real)]
//     real FMAs            = TM * TN * (complex ? 4 : 1)
//
// For COMPLEX<FLOAT> (planes = 2, W = 4) that is 8*TM*TN/(TM+TN) -- exactly
// TWICE the ratio of the real-float kernel at the same tile, because a complex
// MAC costs 4 FMAs while its operands cost only 2x the words.  Concretely:
//
//     TM x TN = 8x8  ->  32:1   (128 accumulator registers)
//     TM x TN = 8x4  ->  21.3:1 ( 64 accumulator registers)
//     TM x TN = 4x4  ->  16:1   ( 32 accumulator registers)
//
// The float kernel's measured-fast 16:1 is reached by a 4x4 COMPLEX tile using
// 32 accumulator registers.  That is the load-bearing result of this file: the
// "64 accumulators or bust" framing does not transfer to complex.  All three
// tiles are selectable so a later measurement can pick between a 32:1 ratio at
// 8 warps/SM and a 16:1 ratio at 24 warps/SM.
//
// For DOUBLE (planes = 1, W = 2) the ratio at 8x8 is only 8:1, because a
// 128-bit shared load carries half as many elements.  That is not a problem:
// an RTX 4090 is a consumer part with 2 FP64 lanes per SM (1/64th of FP32), so
// a warp's 64 DFMAs occupy the FP64 pipe for ~1000 SM-cycles while its 8 LDS
// occupy the LSU for ~8.  The shared pipe cannot be the limit for FP64 here at
// any sane tile.  What matters for double is only (a) not spilling and (b)
// enough resident warps to cover DFMA latency -- which argues for a SMALL
// tile, the opposite of the float advice.  Hence the 64x64/4x4 default for the
// two FP64 types.
//
// ===========================================================================
// 2. PLANAR (SPLIT) VS INTERLEAVED COMPLEX IN SHARED MEMORY
// ===========================================================================
// The assigned design is planar: stage the real and imaginary parts of the
// operands into SEPARATE shared planes (sa_re/sa_im, sb_re/sb_im) so that
// everything below the staging step sees plain floats.  This file implements
// planar AND interleaved as one templated kernel, selectable with --layout,
// because the honest analysis does NOT say planar wins outright:
//
//   * LOAD COUNT IS IDENTICAL.  A thread needs TM complex values of A; that is
//     TM*2 reals either way, i.e. TM/CPP 128-bit loads either way (CPP =
//     scalars per 16-byte packet: 2 for complex<float>, 1 for complex<double>).
//     Planar issues 2 loads per band of W elements, interleaved issues 1 load
//     per band of CPP = W/2 elements.  Same total.  The naive "interleaved
//     costs a 2-way bank conflict" claim is only true if you keep the planar
//     band width; with the narrower band interleaved is conflict-free too.
//     So the fragment load is a WASH, and the code below parameterises the
//     band width (EPV) precisely so both stay conflict-free.
//
//   * PLANAR WINS THE B TRANSPOSE.  B must be staged [k][n] from a [n][k]
//     column-major source, so its shared write is an element scatter, not a
//     packet copy.  Scattering single elements into an INTERLEAVED array has
//     an element stride of P 32-bit words (2 for complex<float>, 4 for
//     complex<double>), so lanes varying in n reach only 32/P of the 32 banks:
//     a hard 2-way / 4-way conflict on every B staging store, unavoidable by
//     any lane assignment.  Planar's stride is 1 word and restores all 32.
//
//   * PLANAR LOSES THE A STAGING STORE.  A is staged [k][m] with m contiguous
//     in the source, so interleaved staging is a pure copy: one conflict-free
//     16-byte STS.  Planar must split the packet into a re half and an im
//     half, which is 2 stores, and they are conflict-free only if the compiler
//     merges each adjacent pair into an STS.64.
//
//   * PLANAR USES W-WIDE BANDS INSTEAD OF CPP-WIDE, so it needs half as many
//     band base addresses (and for complex<double> interleaved degenerates to
//     a band ONE element wide).  I expected that to show up as a register
//     saving.  IT DOES NOT, and the effect reverses with the tile -- see the
//     measured numbers below.  Do not repeat the claim.
//
//   * PLANAR IS STRUCTURALLY SAFER: no std::complex is ever instantiated in
//     device code, so operator*'s isnan branch and __mulsc3 call cannot
//     reappear.  (The inner loop writes ar*br - ai*bi and ar*bi + ai*br by
//     hand in both layouts; planar just makes it impossible to regress.)
//
// Net: planar trades one extra shared store per staged packet for the removal
// of a 2-4x conflict on every B staging store.  Both effects are staging-side,
// i.e. amortised over TileK inner steps -- the PTX census below puts the whole
// difference at 16 vs 12 shared stores against 2048 FMAs per k-tile, under 1%
// of issue either way.  The sign is NOT predictable from analysis, which is
// why both layouts are built rather than argued about.
//
// ===========================================================================
// 2b. WHAT WAS ACTUALLY MEASURED AT COMPILE TIME (sm_89, -Xcuda-ptxas -v)
// ===========================================================================
// Registers / spill bytes, unpredicated path:
//
//   dtype    tile            planar        interleaved
//   cfloat   128x128x8/8x8   247, 0 spill  210, 0 spill
//   cfloat   128x128x8/8x4   122, 0 spill  126, 0 spill
//   cfloat   64x64x8/4x4      72, 0 spill   80, 0 spill
//   cdouble  128x128x8/8x8   255, 3444 B   255, 3304 B   <-- both spill
//   cdouble  128x128x8/8x4   208, 0 spill  226, 0 spill
//   cdouble  64x64x8/4x4     138, 0 spill  126, 0 spill
//   double   128x128x8/8x8   208, 0 spill  (n/a)
//   double   64x64x8/4x4      72, 0 spill  (n/a)
//   float    128x128x8/8x8   117, 0 spill  (n/a)
//
// Two of those contradict the received wisdom that a 64-accumulator thread
// tile "cannot fit and spills" for anything but float:
//   * complex<float> at 8x8 is 128 accumulator registers and DOES fit, at 247
//     of 255 with zero spill -- but with an 8-register margin, and at
//     256 thr x 247 = 63232 of the SM's 65536 registers, i.e. ONE block and 8
//     warps per SM.  It is a real configuration and a fragile one.
//   * double at 8x8 (also 128 accumulator registers) fits comfortably at 208.
// Only complex<double> at 8x8 (256 accumulator registers) genuinely cannot,
// and it spills ~3.4 KB in both layouts.  That one is kept selectable only so
// the cliff can be measured rather than assumed.
//
// PTX census per kernel (fast path, per k-tile of the main loop):
//
//   dtype    tile            128-bit shared loads   inner FMAs   ratio
//   cfloat   128x128x8/8x8          64                 2048      32.0:1
//   cfloat   128x128x8/8x4          48                 1024      21.3:1
//   cfloat   64x64x8/4x4            32                  512      16.0:1
//   cdouble  64x64x8/4x4            64                  512       8.0:1
//   double   128x128x8/8x8          64                  512       8.0:1
//   float    128x128x8/8x8          32                  512      16.0:1   <-- the in-tree kernel's ratio
//
// and, across all 48 kernels in this file, the count of SCALAR shared loads is
// ZERO: every one is ld.shared.v4.b32 (32-bit scalars) or ld.shared.v2.b64
// (64-bit scalars), both 128-bit forms.  Likewise every fast-path C access is
// a 128-bit st.global / ld.global.  There are no call.uni and no __mulsc3 /
// __muldc3 anywhere.
//
// ===========================================================================
// 3. WHAT IS PRESERVED FROM THE FLOAT KERNEL
// ===========================================================================
//   * Shared strides exactly TileM / TileN per plane, never TileM+1.  An odd
//     stride means the compiler cannot prove 16-byte alignment and every
//     fragment load degrades to scalar ld.shared.b32.
//   * B staged [k][n], never [n][k], so a thread's n-fragment is contiguous.
//   * The thread's rows/cols are split into TM/EPV bands of EPV contiguous
//     elements, spaced TileM/NB apart, so each 128-bit load is serviced
//     8 lanes x 16 B = exactly the 32 banks.  Note EPV is derived from a BYTE
//     constant: the original kernel's Packet4<T> is 32 bytes for double and 64
//     for complex<double> -- accesses no shared-load instruction implements,
//     demanding an alignment nothing needs.  16 bytes is the real hardware
//     width (ld.shared.v4.f32 and ld.shared.v2.f64 are each one instruction).
//   * The m index is the FASTEST-VARYING thread index (local_id(2)), so the
//     epilogue's read-modify-write of a column-major C is coalesced.  Getting
//     this backwards is free at beta == 0 and catastrophic at beta != 0; this
//     harness therefore defaults to beta = 1.
//   * Accumulators are plain local arrays; nothing is reachable by reference.
//
// Build:
//   /opt/dpcpp-cuda/bin/clang++ -O3 -std=c++20 -fsycl \
//       -fsycl-targets=nvidia_gpu_sm_89 --cuda-path=/usr/local/cuda-13.2 \
//       -Xcuda-ptxas -v tile-complex-split.cpp -o tile-complex-split
//
// Correctness is checked by default before any timing: the fast and the
// predicated path on the same aligned shape, plus a ragged shape, plus the
// timed shape itself when it is small enough to reference on the CPU.
// verify_all-complex-split.sh runs the whole dtype x tile x layout x beta
// matrix on a SYCL CPU device (--device cpu), so the logic can be validated
// without a GPU; all 48 combinations pass at roundoff.
//
// Measure with e.g.
//   ./tile-complex-split --dtype cfloat --tile 128x128x8/8x8 --layout planar \
//       --m 512 --n 512 --k 512 --batch 512 --beta 1
// beta defaults to 1 deliberately: the epilogue's scattered-read bug that cost
// the float kernel 26 vs 41 TFLOP/s is invisible at beta = 0.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ===========================================================================
// scalar traits and the 16-byte packet
// ===========================================================================

template <typename T>
struct ScalarTraits {
    using real = T;
    static constexpr bool is_complex = false;
    static constexpr int planes = 1;
};

template <typename R>
struct ScalarTraits<std::complex<R>> {
    using real = R;
    static constexpr bool is_complex = true;
    static constexpr int planes = 2;
};

// 16 bytes: the widest load/store form the hardware actually has.
// ld.shared.v4.f32 and ld.shared.v2.f64 are one instruction each;
// ld.shared.v4.f64 is not an instruction at all.
template <typename R, int W>
struct alignas(16) Pkt {
    R v[W];
};

template <int W, typename R>
static inline Pkt<R, W>& pkt(R* p) {
    return *reinterpret_cast<Pkt<R, W>*>(p);
}

template <int W, typename R>
static inline const Pkt<R, W>& pkt(const R* p) {
    return *reinterpret_cast<const Pkt<R, W>*>(p);
}

// ===========================================================================
// tile configuration
// ===========================================================================

template <int TileM_, int TileN_, int TileK_, int ThrM_, int ThrN_>
struct TileCfg {
    static constexpr int TileM = TileM_;
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;
    static constexpr int ThrM = ThrM_;
    static constexpr int ThrN = ThrN_;
};

using Cfg128_8x8 = TileCfg<128, 128, 8, 8, 8>;
using Cfg128_8x4 = TileCfg<128, 128, 8, 8, 4>;
using Cfg64_4x4 = TileCfg<64, 64, 8, 4, 4>;
using Cfg64K16_4x4 = TileCfg<64, 64, 16, 4, 4>;

struct CfgFacts {
    int threads;
    int accum_regs;   // real-valued accumulator registers per thread
    double ratio;     // real FMAs per 128-bit shared load, per k-step
    int shared_bytes;
};

template <typename T, typename Cfg>
static CfgFacts cfg_facts() {
    using ST = ScalarTraits<T>;
    using R = typename ST::real;
    constexpr int P = ST::planes;
    constexpr int W = 16 / (int)sizeof(R);
    const int threads = (Cfg::TileM / Cfg::ThrM) * (Cfg::TileN / Cfg::ThrN);
    const int accum = Cfg::ThrM * Cfg::ThrN * P * (int)(sizeof(R) / sizeof(float));
    const double loads = (double)(Cfg::ThrM + Cfg::ThrN) * P / W;
    const double fmas = (double)Cfg::ThrM * Cfg::ThrN * (P == 2 ? 4 : 1);
    const int shared =
        (int)((Cfg::TileK * Cfg::TileM + Cfg::TileK * Cfg::TileN) * P * sizeof(R));
    return CfgFacts{threads, accum, fmas / loads, shared};
}

// ===========================================================================
// the kernel
// ===========================================================================

template <typename T, typename Cfg, bool Predicated, bool Planar>
class SplitGemmKernel;

// Batched C = alpha * A * B + beta * C, all column-major, NN only.
//
// Predicated == false is the unpredicated fast path; it requires
// m % TileM == 0, n % TileN == 0, k % TileK == 0, 16-byte aligned bases and
// leading dimensions / batch strides that are multiples of CPP elements.
// Predicated == true is correct for any shape: it zero-fills the shared tile,
// so the inner loop is bit-identical between the two paths.
//
// Planar == true stages complex operands as two real planes; Planar == false
// keeps them interleaved.  For real scalars the two are the same code and only
// the planar instantiation is ever created.
template <typename T, typename Cfg, bool Predicated, bool Planar>
static sycl::event launch_split_gemm(sycl::queue& q,
                                     const T* Ap, int lda, long long stride_a,
                                     const T* Bp, int ldb, long long stride_b,
                                     T* Cp, int ldc, long long stride_c,
                                     int m, int n, int k, int batch,
                                     T alpha, T beta) {
    using ST = ScalarTraits<T>;
    using R = typename ST::real;

    constexpr bool IsC = ST::is_complex;
    constexpr int P = ST::planes;                 // reals per scalar
    constexpr int W = 16 / (int)sizeof(R);        // reals per 128-bit packet
    constexpr int CPP = W / P;                    // scalars per 128-bit packet
    constexpr bool Split = IsC && Planar;

    // Scalars delivered by one 128-bit shared load, and therefore the width of
    // one band of the thread tile.  Planar reads W elements of one plane;
    // interleaved reads CPP elements of both parts.  Either way a band is
    // exactly 16 bytes per lane, which is what keeps 8 lanes x 16 B = 32 banks.
    constexpr int EPV = Split ? W : CPP;
    // Element -> real-index scale inside the shared tile.
    constexpr int SC = Split ? 1 : P;

    constexpr int TileM = Cfg::TileM;
    constexpr int TileN = Cfg::TileN;
    constexpr int TileK = Cfg::TileK;
    constexpr int ThrM = Cfg::ThrM;
    constexpr int ThrN = Cfg::ThrN;

    constexpr int LM = TileM / ThrM;              // threads down m
    constexpr int LN = TileN / ThrN;              // threads across n
    constexpr int Threads = LM * LN;

    constexpr int NBM = ThrM / EPV;               // m bands per thread
    constexpr int NBN = ThrN / EPV;               // n bands per thread
    constexpr int SpanM = TileM / NBM;            // distance between m bands
    constexpr int SpanN = TileN / NBN;

    static_assert(W * (int)sizeof(R) == 16, "packet must be 16 bytes");
    static_assert(W % P == 0, "a packet must hold a whole number of scalars");
    static_assert(TileM % ThrM == 0 && TileN % ThrN == 0, "tile/thread mismatch");
    static_assert(ThrM % EPV == 0 && ThrN % EPV == 0, "thread tile must be whole bands");
    static_assert(EPV % CPP == 0, "a band must be a whole number of C packets");
    static_assert(LM * EPV * NBM == TileM, "m banding does not tile TileM");
    static_assert(LN * EPV * NBN == TileN, "n banding does not tile TileN");
    static_assert(TileM % CPP == 0 && TileK % CPP == 0, "staging granularity");
    static_assert(Threads <= 1024, "too many threads per group");
    static_assert((TileK * TileM + TileK * TileN) * P * (int)sizeof(R) <= 48 * 1024,
                  "shared tile exceeds the 48 KB static limit");

    // Shared tiles.  Stride is exactly TileM / TileN elements, no padding: an
    // aligned stride is the whole reason the fragment loads vectorise.
    constexpr int ASz = TileK * TileM;            // elements
    constexpr int BSz = TileK * TileN;

    // Staging: every thread moves whole 16-byte packets.
    constexpr int AMPk = TileM / CPP;             // A packets per k row
    constexpr int APkts = AMPk * TileK;
    constexpr int APer = (APkts + Threads - 1) / Threads;
    constexpr int BKPk = TileK / CPP;             // B packets per n column
    constexpr int BPkts = BKPk * TileN;
    constexpr int BPer = (BPkts + Threads - 1) / Threads;

    const int gM = (m + TileM - 1) / TileM;
    const int gN = (n + TileN - 1) / TileN;

    // Everything crossing into device code is a real scalar or a raw pointer.
    // std::complex is never instantiated on the device.
    const R* Ar = reinterpret_cast<const R*>(Ap);
    const R* Br = reinterpret_cast<const R*>(Bp);
    R* Cr = reinterpret_cast<R*>(Cp);

    R al_re, al_im, be_re, be_im;
    if constexpr (IsC) {
        al_re = alpha.real();
        al_im = alpha.imag();
        be_re = beta.real();
        be_im = beta.imag();
    } else {
        al_re = alpha;
        al_im = R(0);
        be_re = beta;
        be_im = R(0);
    }
    const bool beta_nz = (be_re != R(0)) || (be_im != R(0));

    // dim 0 = batch, dim 1 = n tiles, dim 2 = m tiles.  local_id(2) is the
    // fastest-varying SYCL dimension (it maps to threadIdx.x) and it is bound
    // to m -- see the epilogue note in section 3 of the file header.
    const sycl::range<3> local(1, LN, LM);
    const sycl::range<3> global((size_t)batch, (size_t)gN * LN, (size_t)gM * LM);

    return q.submit([&](sycl::handler& h) {
        // Allocated as packets so the base is guaranteed 16-byte aligned; a
        // local_accessor<double,1> is only guaranteed 8-byte aligned, which
        // would silently break every ld.shared.v2.f64.
        sycl::local_accessor<Pkt<R, W>, 1> smem(sycl::range<1>(((ASz + BSz) * P) / W), h);

        h.parallel_for<SplitGemmKernel<T, Cfg, Predicated, Planar>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> it) {
                const int bid = (int)it.get_group(0);

                const int ty = (int)it.get_local_id(2);   // 0..LM-1, m
                const int tx = (int)it.get_local_id(1);   // 0..LN-1, n
                const int tid = tx * LM + ty;             // = linear local id

                const int n0 = (int)it.get_group(1) * TileN;
                const int m0 = (int)it.get_group(2) * TileM;

                R* base = reinterpret_cast<R*>(
                    smem.template get_multi_ptr<sycl::access::decorated::no>().get());
                // Planar: [a_re][a_im][b_re][b_im].  Interleaved: [A][B], and
                // the _im pointers are never used.
                R* sa_re = base;
                R* sa_im = base + (Split ? ASz : 0);
                R* sb_re = base + ASz * P;
                R* sb_im = sb_re + (Split ? BSz : 0);

                const R* Ab = Ar + (std::ptrdiff_t)bid * stride_a * P;
                const R* Bb = Br + (std::ptrdiff_t)bid * stride_b * P;
                R* Cb = Cr + (std::ptrdiff_t)bid * stride_c * P;

                // Plain local arrays.  An accumulator reachable through a
                // reference has been measured to spill in this codebase.
                R acc_re[ThrM][ThrN];
                R acc_im[IsC ? ThrM : 1][IsC ? ThrN : 1];
#pragma unroll
                for (int i = 0; i < ThrM; ++i) {
#pragma unroll
                    for (int j = 0; j < ThrN; ++j) {
                        acc_re[i][j] = R(0);
                    }
                }
                if constexpr (IsC) {
#pragma unroll
                    for (int i = 0; i < ThrM; ++i) {
#pragma unroll
                        for (int j = 0; j < ThrN; ++j) {
                            acc_im[i][j] = R(0);
                        }
                    }
                }

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    // ------------------------------------------------------
                    // stage A as [k][m]
                    // ------------------------------------------------------
#pragma unroll
                    for (int p = 0; p < APer; ++p) {
                        const int pid = tid + p * Threads;
                        if constexpr (APkts % Threads != 0) {
                            if (pid >= APkts) continue;
                        }
                        const int am = (pid % AMPk) * CPP;
                        const int ak = pid / AMPk;
                        const int so = (ak * TileM + am) * SC;
                        if constexpr (!Predicated) {
                            // One coalesced 16-byte load per lane: consecutive
                            // tid are consecutive m in a column-major A.
                            const Pkt<R, W> v = pkt<W>(
                                Ab + (std::ptrdiff_t)P *
                                         ((m0 + am) + (std::ptrdiff_t)(k0 + ak) * lda));
                            if constexpr (Split) {
#pragma unroll
                                for (int e = 0; e < CPP; ++e) {
                                    sa_re[so + e] = v.v[2 * e];
                                    sa_im[so + e] = v.v[2 * e + 1];
                                }
                            } else {
                                // Interleaved staging of A is a pure copy.
                                pkt<W>(&sa_re[so]) = v;
                            }
                        } else {
                            const int gk = k0 + ak;
#pragma unroll
                            for (int e = 0; e < CPP; ++e) {
                                const int gm = m0 + am + e;
                                const bool live = (gm < m) && (gk < k);
                                const std::ptrdiff_t off =
                                    (std::ptrdiff_t)P * (gm + (std::ptrdiff_t)gk * lda);
                                if constexpr (Split) {
                                    sa_re[so + e] = live ? Ab[off] : R(0);
                                    if constexpr (IsC) {
                                        sa_im[so + e] = live ? Ab[off + 1] : R(0);
                                    }
                                } else {
                                    sa_re[so + e * P] = live ? Ab[off] : R(0);
                                    if constexpr (IsC) {
                                        sa_re[so + e * P + 1] = live ? Ab[off + 1] : R(0);
                                    }
                                }
                            }
                        }
                    }
                    // ------------------------------------------------------
                    // stage B as [k][n] -- transposed on the store side
                    // ------------------------------------------------------
#pragma unroll
                    for (int p = 0; p < BPer; ++p) {
                        const int pid = tid + p * Threads;
                        if constexpr (BPkts % Threads != 0) {
                            if (pid >= BPkts) continue;
                        }
                        const int bk = (pid % BKPk) * CPP;
                        const int bn = pid / BKPk;
                        if constexpr (!Predicated) {
                            const Pkt<R, W> v = pkt<W>(
                                Bb + (std::ptrdiff_t)P *
                                         ((k0 + bk) + (std::ptrdiff_t)(n0 + bn) * ldb));
#pragma unroll
                            for (int e = 0; e < CPP; ++e) {
                                const int so = ((bk + e) * TileN + bn) * SC;
                                if constexpr (Split) {
                                    sb_re[so] = v.v[2 * e];
                                    sb_im[so] = v.v[2 * e + 1];
                                } else if constexpr (IsC) {
                                    sb_re[so] = v.v[2 * e];
                                    sb_re[so + 1] = v.v[2 * e + 1];
                                } else {
                                    sb_re[so] = v.v[e];
                                }
                            }
                        } else {
                            const int gn = n0 + bn;
#pragma unroll
                            for (int e = 0; e < CPP; ++e) {
                                const int gk = k0 + bk + e;
                                const bool live = (gk < k) && (gn < n);
                                const std::ptrdiff_t off =
                                    (std::ptrdiff_t)P * (gk + (std::ptrdiff_t)gn * ldb);
                                const int so = ((bk + e) * TileN + bn) * SC;
                                if constexpr (Split) {
                                    sb_re[so] = live ? Bb[off] : R(0);
                                    if constexpr (IsC) {
                                        sb_im[so] = live ? Bb[off + 1] : R(0);
                                    }
                                } else {
                                    sb_re[so] = live ? Bb[off] : R(0);
                                    if constexpr (IsC) {
                                        sb_re[so + 1] = live ? Bb[off + 1] : R(0);
                                    }
                                }
                            }
                        }
                    }

                    it.barrier(sycl::access::fence_space::local_space);

                    // ------------------------------------------------------
                    // inner loop: plain reals, four explicit FMA chains
                    // ------------------------------------------------------
#pragma unroll
                    for (int kk = 0; kk < TileK; ++kk) {
                        R af_re[ThrM], bf_re[ThrN];
                        R af_im[IsC ? ThrM : 1], bf_im[IsC ? ThrN : 1];
#pragma unroll
                        for (int b = 0; b < NBM; ++b) {
                            const int o = (kk * TileM + b * SpanM + ty * EPV) * SC;
                            const Pkt<R, W> t = pkt<W>(&sa_re[o]);
                            if constexpr (Split) {
#pragma unroll
                                for (int e = 0; e < W; ++e) af_re[b * EPV + e] = t.v[e];
                                const Pkt<R, W> u = pkt<W>(&sa_im[o]);
#pragma unroll
                                for (int e = 0; e < W; ++e) af_im[b * EPV + e] = u.v[e];
                            } else {
#pragma unroll
                                for (int e = 0; e < EPV; ++e) {
                                    af_re[b * EPV + e] = t.v[e * P];
                                    if constexpr (IsC) af_im[b * EPV + e] = t.v[e * P + 1];
                                }
                            }
                        }
#pragma unroll
                        for (int b = 0; b < NBN; ++b) {
                            const int o = (kk * TileN + b * SpanN + tx * EPV) * SC;
                            const Pkt<R, W> t = pkt<W>(&sb_re[o]);
                            if constexpr (Split) {
#pragma unroll
                                for (int e = 0; e < W; ++e) bf_re[b * EPV + e] = t.v[e];
                                const Pkt<R, W> u = pkt<W>(&sb_im[o]);
#pragma unroll
                                for (int e = 0; e < W; ++e) bf_im[b * EPV + e] = u.v[e];
                            } else {
#pragma unroll
                                for (int e = 0; e < EPV; ++e) {
                                    bf_re[b * EPV + e] = t.v[e * P];
                                    if constexpr (IsC) bf_im[b * EPV + e] = t.v[e * P + 1];
                                }
                            }
                        }
#pragma unroll
                        for (int i = 0; i < ThrM; ++i) {
#pragma unroll
                            for (int j = 0; j < ThrN; ++j) {
                                if constexpr (IsC) {
                                    // Written out by hand.  std::complex's
                                    // operator* would emit an isnan branch and
                                    // a call to __mulsc3 right here.
                                    acc_re[i][j] += af_re[i] * bf_re[j];
                                    acc_re[i][j] -= af_im[i] * bf_im[j];
                                    acc_im[i][j] += af_re[i] * bf_im[j];
                                    acc_im[i][j] += af_im[i] * bf_re[j];
                                } else {
                                    acc_re[i][j] += af_re[i] * bf_re[j];
                                }
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // ----------------------------------------------------------
                // epilogue.  Within a band the EPV rows are consecutive in m,
                // the contiguous direction of a column-major C, and m is the
                // fastest-varying thread index, so a band is EPV/CPP aligned
                // 128-bit accesses and lanes of a warp touch adjacent
                // addresses.  That matters only when beta != 0, which is why
                // this harness defaults to beta = 1.
                // ----------------------------------------------------------
                if constexpr (!Predicated) {
                    constexpr int Chunks = EPV / CPP;   // 16-byte stores per band
#pragma unroll
                    for (int j = 0; j < ThrN; ++j) {
                        const int gn = n0 + (j / EPV) * SpanN + tx * EPV + (j % EPV);
#pragma unroll
                        for (int b = 0; b < NBM; ++b) {
#pragma unroll
                            for (int c = 0; c < Chunks; ++c) {
                                const int gm = m0 + b * SpanM + ty * EPV + c * CPP;
                                R* q0 = Cb + (std::ptrdiff_t)P *
                                                 (gm + (std::ptrdiff_t)gn * ldc);
                                Pkt<R, W> out;
                                Pkt<R, W> prior;
                                if (beta_nz) prior = pkt<W>((const R*)q0);
#pragma unroll
                                for (int e = 0; e < CPP; ++e) {
                                    const int i = b * EPV + c * CPP + e;
                                    if constexpr (IsC) {
                                        const R xr = acc_re[i][j];
                                        const R xi = acc_im[i][j];
                                        R yr = al_re * xr - al_im * xi;
                                        R yi = al_re * xi + al_im * xr;
                                        if (beta_nz) {
                                            const R pr = prior.v[2 * e];
                                            const R pi = prior.v[2 * e + 1];
                                            yr += be_re * pr - be_im * pi;
                                            yi += be_re * pi + be_im * pr;
                                        }
                                        out.v[2 * e] = yr;
                                        out.v[2 * e + 1] = yi;
                                    } else {
                                        R y = al_re * acc_re[i][j];
                                        if (beta_nz) y += be_re * prior.v[e];
                                        out.v[e] = y;
                                    }
                                }
                                pkt<W>(q0) = out;
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < ThrN; ++j) {
                        const int gn = n0 + (j / EPV) * SpanN + tx * EPV + (j % EPV);
                        if (gn >= n) continue;
#pragma unroll
                        for (int i = 0; i < ThrM; ++i) {
                            const int gm = m0 + (i / EPV) * SpanM + ty * EPV + (i % EPV);
                            if (gm >= m) continue;
                            R* q0 = Cb + (std::ptrdiff_t)P *
                                             (gm + (std::ptrdiff_t)gn * ldc);
                            if constexpr (IsC) {
                                const R xr = acc_re[i][j];
                                const R xi = acc_im[i][j];
                                R yr = al_re * xr - al_im * xi;
                                R yi = al_re * xi + al_im * xr;
                                if (beta_nz) {
                                    const R pr = q0[0];
                                    const R pi = q0[1];
                                    yr += be_re * pr - be_im * pi;
                                    yi += be_re * pi + be_im * pr;
                                }
                                q0[0] = yr;
                                q0[1] = yi;
                            } else {
                                R y = al_re * acc_re[i][j];
                                if (beta_nz) y += be_re * q0[0];
                                q0[0] = y;
                            }
                        }
                    }
                }
            });
    });
}

// ===========================================================================
// host side
// ===========================================================================

template <typename T>
struct AccOf {
    using type = double;
};
template <>
struct AccOf<std::complex<float>> {
    using type = std::complex<double>;
};
template <>
struct AccOf<std::complex<double>> {
    using type = std::complex<double>;
};

template <typename T>
static typename AccOf<T>::type to_acc(const T& x) {
    if constexpr (ScalarTraits<T>::is_complex) {
        return std::complex<double>((double)x.real(), (double)x.imag());
    } else {
        return (double)x;
    }
}

template <typename T>
static T from_acc(const typename AccOf<T>::type& x) {
    using R = typename ScalarTraits<T>::real;
    if constexpr (ScalarTraits<T>::is_complex) {
        return T((R)x.real(), (R)x.imag());
    } else {
        return (T)x;
    }
}

template <typename T>
static double magnitude(const T& x) {
    if constexpr (ScalarTraits<T>::is_complex) {
        return std::hypot((double)x.real(), (double)x.imag());
    } else {
        return std::fabs((double)x);
    }
}

// Deterministic fill, same generator family as experiments/sycl_vs_cuda.
template <typename T>
static void fill(std::vector<T>& v, unsigned a, unsigned c) {
    using R = typename ScalarTraits<T>::real;
    for (size_t i = 0; i < v.size(); ++i) {
        const R re = (R)((double)((i * a + c) % 1000) / 1000.0 - 0.5);
        if constexpr (ScalarTraits<T>::is_complex) {
            const R im = (R)((double)(((i * 2 + 7) * a + c) % 1000) / 1000.0 - 0.5);
            v[i] = T(re, im);
        } else {
            v[i] = re;
        }
    }
}

template <typename T>
static void reference_gemm(int m, int n, int k, int batch,
                           const std::vector<T>& A, int lda, long long sa,
                           const std::vector<T>& B, int ldb, long long sb,
                           const std::vector<T>& Cin, int ldc, long long sc,
                           T alpha, T beta, std::vector<T>& out) {
    using Acc = typename AccOf<T>::type;
    const Acc al = to_acc(alpha);
    const Acc be = to_acc(beta);
    out.resize(Cin.size());
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                Acc s = Acc(0);
                for (int kk = 0; kk < k; ++kk) {
                    s += to_acc(A[(size_t)(b * sa) + (size_t)kk * lda + i]) *
                         to_acc(B[(size_t)(b * sb) + (size_t)j * ldb + kk]);
                }
                const size_t ci = (size_t)(b * sc) + (size_t)j * ldc + i;
                out[ci] = from_acc<T>(al * s + be * to_acc(Cin[ci]));
            }
        }
    }
}

template <typename T>
static double compare(const std::vector<T>& got, const std::vector<T>& ref) {
    double scale = 0.0;
    for (const auto& r : ref) scale += magnitude(r);
    scale = ref.empty() ? 1.0 : scale / (double)ref.size();
    if (scale <= 0.0) scale = 1.0;
    double worst = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = magnitude(T(got[i] - ref[i]));
        worst = std::max(worst, d / (magnitude(ref[i]) + scale));
    }
    return worst;
}

// --- config dispatch -------------------------------------------------------

template <typename T>
using LaunchFn = sycl::event (*)(sycl::queue&, const T*, int, long long, const T*, int,
                                 long long, T*, int, long long, int, int, int, int, T, T);

struct CfgEntry {
    const char* name;
    int TileM, TileN, TileK, ThrM, ThrN;
};

static const CfgEntry kCfgs[] = {
    {"128x128x8/8x8", 128, 128, 8, 8, 8},
    {"128x128x8/8x4", 128, 128, 8, 8, 4},
    {"64x64x8/4x4", 64, 64, 8, 4, 4},
    {"64x64x16/4x4", 64, 64, 16, 4, 4},
};

template <typename T, bool Pred, bool Planar>
static LaunchFn<T> pick_cfg(const std::string& name) {
    if (name == kCfgs[0].name) return &launch_split_gemm<T, Cfg128_8x8, Pred, Planar>;
    if (name == kCfgs[1].name) return &launch_split_gemm<T, Cfg128_8x4, Pred, Planar>;
    if (name == kCfgs[2].name) return &launch_split_gemm<T, Cfg64_4x4, Pred, Planar>;
    if (name == kCfgs[3].name) return &launch_split_gemm<T, Cfg64K16_4x4, Pred, Planar>;
    return nullptr;
}

// For real scalars the interleaved and planar bodies are identical code, so
// only the planar instantiation is ever created.
template <typename T, bool Pred>
static LaunchFn<T> pick(const std::string& name, bool planar) {
    if constexpr (ScalarTraits<T>::is_complex) {
        return planar ? pick_cfg<T, Pred, true>(name) : pick_cfg<T, Pred, false>(name);
    } else {
        return pick_cfg<T, Pred, true>(name);
    }
}

template <typename T>
static CfgFacts facts_of(const std::string& name) {
    if (name == kCfgs[0].name) return cfg_facts<T, Cfg128_8x8>();
    if (name == kCfgs[1].name) return cfg_facts<T, Cfg128_8x4>();
    if (name == kCfgs[2].name) return cfg_facts<T, Cfg64_4x4>();
    return cfg_facts<T, Cfg64K16_4x4>();
}

static const CfgEntry& entry_of(const std::string& name) {
    for (const auto& e : kCfgs)
        if (name == e.name) return e;
    return kCfgs[0];
}

// --- one correctness run ---------------------------------------------------

template <typename T>
static double check_one(sycl::queue& q, const std::string& cfg, bool planar,
                        bool predicated, int m, int n, int k, int batch) {
    using R = typename ScalarTraits<T>::real;
    const int lda = m, ldb = k, ldc = m;
    const long long sa = (long long)m * k, sb = (long long)k * n, sc = (long long)m * n;

    std::vector<T> hA((size_t)sa * batch), hB((size_t)sb * batch), hC((size_t)sc * batch);
    fill(hA, 1103515245u, 12345u);
    fill(hB, 22695477u, 1u);
    fill(hC, 69069u, 5u);

    // Deliberately complex alpha and beta, so both terms of the epilogue's
    // hand-written complex multiply are exercised.
    T alpha, beta;
    if constexpr (ScalarTraits<T>::is_complex) {
        alpha = T((R)1.25, (R)-0.5);
        beta = T((R)0.75, (R)0.25);
    } else {
        alpha = (T)1.25;
        beta = (T)0.75;
    }

    T* dA = sycl::malloc_device<T>(hA.size(), q);
    T* dB = sycl::malloc_device<T>(hB.size(), q);
    T* dC = sycl::malloc_device<T>(hC.size(), q);
    q.memcpy(dA, hA.data(), hA.size() * sizeof(T)).wait();
    q.memcpy(dB, hB.data(), hB.size() * sizeof(T)).wait();
    q.memcpy(dC, hC.data(), hC.size() * sizeof(T)).wait();

    LaunchFn<T> fn = predicated ? pick<T, true>(cfg, planar) : pick<T, false>(cfg, planar);
    fn(q, dA, lda, sa, dB, ldb, sb, dC, ldc, sc, m, n, k, batch, alpha, beta).wait();

    std::vector<T> got(hC.size());
    q.memcpy(got.data(), dC, hC.size() * sizeof(T)).wait();
    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);

    std::vector<T> ref;
    reference_gemm(m, n, k, batch, hA, lda, sa, hB, ldb, sb, hC, ldc, sc, alpha, beta, ref);
    const double e = compare(got, ref);
    std::fprintf(stderr, "  check %-14s %-11s %-10s m=%d n=%d k=%d batch=%d -> maxrelerr %.3e\n",
                 cfg.c_str(), planar ? "planar" : "interleaved",
                 predicated ? "predicated" : "fast", m, n, k, batch, e);
    return e;
}

struct Opts {
    int m = 512, n = 512, k = 512, batch = 512;
    int iters = 30, warmup = 10;
    double beta = 1.0;
    std::string dtype = "cfloat";
    std::string tile;
    std::string device = "gpu";
    std::string layout = "planar";
    bool check = true;
};

// --device cpu selects a SYCL CPU device.  It exists so the correctness check
// can be run without a GPU: the index arithmetic, the planar split and the
// epilogue are the same source text on either device, so a CPU pass is a real
// statement about the kernel's logic (it says nothing about its speed).
static sycl::queue make_queue(const std::string& which) {
    if (which == "cpu") return sycl::queue{sycl::cpu_selector_v};
    if (which == "any") return sycl::queue{sycl::default_selector_v};
    return sycl::queue{sycl::gpu_selector_v};
}

template <typename T>
static int run(const Opts& o) {
    using ST = ScalarTraits<T>;
    using R = typename ST::real;

    sycl::queue q = make_queue(o.device);
    std::fprintf(stderr, "device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    const bool planar = (o.layout != "interleaved");
    if (!ST::is_complex && !planar) {
        std::fprintf(stderr,
                     "note: --layout interleaved is meaningless for a real scalar "
                     "(the two layouts are the same code); using planar\n");
    }

    std::string cfg = o.tile;
    if (cfg.empty()) {
        // Defaults follow the argument in the file header: complex<float> is
        // FMA-rich enough to want the big thread tile; the FP64 types are
        // limited by a 2-lane-per-SM FP64 pipe and want registers instead.
        cfg = (sizeof(R) == 4) ? "128x128x8/8x8" : "64x64x8/4x4";
    }
    if (!pick<T, false>(cfg, planar)) {
        std::fprintf(stderr, "unknown tile '%s'; known:", cfg.c_str());
        for (const auto& e : kCfgs) std::fprintf(stderr, " %s", e.name);
        std::fprintf(stderr, "\n");
        return 2;
    }
    const CfgEntry& ce = entry_of(cfg);
    const CfgFacts cf = facts_of<T>(cfg);
    std::fprintf(stderr,
                 "tile %s (%s): %d threads/group, %d accumulator regs/thread, "
                 "%.1f:1 real-FMA per 128-bit shared load, %d B shared\n",
                 cfg.c_str(), planar ? "planar" : "interleaved", cf.threads,
                 cf.accum_regs, cf.ratio, cf.shared_bytes);

    double maxrelerr = -1.0;
    if (o.check) {
        // Aligned shape, both paths on identical data: a divergence pins the
        // bug to the staging or the epilogue rather than to the inner loop.
        maxrelerr = std::max(
            maxrelerr, check_one<T>(q, cfg, planar, false, ce.TileM, ce.TileN, 2 * ce.TileK, 3));
        maxrelerr = std::max(
            maxrelerr, check_one<T>(q, cfg, planar, true, ce.TileM, ce.TileN, 2 * ce.TileK, 3));
        // Ragged shape: only the predicated path is valid here.
        maxrelerr = std::max(
            maxrelerr,
            check_one<T>(q, cfg, planar, true, ce.TileM + 3, ce.TileN - 7, ce.TileK + 5, 2));
        if (maxrelerr > 1e-4) {
            std::fprintf(stderr, "CORRECTNESS FAILURE: maxrelerr %.3e\n", maxrelerr);
            return 3;
        }
    }

    const int m = o.m, n = o.n, k = o.k, batch = o.batch;
    const int lda = m, ldb = k, ldc = m;
    const long long sa = (long long)m * k, sb = (long long)k * n, sc = (long long)m * n;
    const bool fast = (m % ce.TileM == 0) && (n % ce.TileN == 0) && (k % ce.TileK == 0);

    std::vector<T> hA((size_t)sa * batch), hB((size_t)sb * batch), hC((size_t)sc * batch);
    fill(hA, 1103515245u, 12345u);
    fill(hB, 22695477u, 1u);
    fill(hC, 69069u, 5u);   // C is always initialised; beta != 0 must read it.

    T* dA = sycl::malloc_device<T>(hA.size(), q);
    T* dB = sycl::malloc_device<T>(hB.size(), q);
    T* dC = sycl::malloc_device<T>(hC.size(), q);
    if (!dA || !dB || !dC) {
        std::fprintf(stderr, "device allocation failed\n");
        return 4;
    }
    q.memcpy(dA, hA.data(), hA.size() * sizeof(T)).wait();
    q.memcpy(dB, hB.data(), hB.size() * sizeof(T)).wait();
    q.memcpy(dC, hC.data(), hC.size() * sizeof(T)).wait();

    T alpha, beta;
    if constexpr (ST::is_complex) {
        alpha = T((R)1, (R)0);
        beta = T((R)o.beta, (R)0);
    } else {
        alpha = (T)1;
        beta = (T)o.beta;
    }

    // The fast path is unpredicated and has no runtime alignment check of its
    // own: it assumes 16-byte aligned bases and leading dimensions / batch
    // strides that are a whole number of 16-byte packets.  This harness's own
    // allocation satisfies that, but assert it rather than assume it.
    constexpr int PacketElems = 16 / (int)sizeof(T);
    auto ok16 = [](const void* p) {
        return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % 16) == 0;
    };
    const bool aligned = ok16(dA) && ok16(dB) && ok16(dC) &&
                         (lda % PacketElems == 0) && (ldb % PacketElems == 0) &&
                         (ldc % PacketElems == 0) && (sa % PacketElems == 0) &&
                         (sb % PacketElems == 0) && (sc % PacketElems == 0);
    const bool use_fast = fast && aligned;

    LaunchFn<T> fn =
        use_fast ? pick<T, false>(cfg, planar) : pick<T, true>(cfg, planar);
    if (!use_fast) {
        std::fprintf(stderr,
                     "note: timing the predicated path (%s)\n",
                     !fast ? "shape is not a multiple of the tile"
                           : "operands do not meet the fast path's alignment contract");
    }

    auto go = [&]() {
        return fn(q, dA, lda, sa, dB, ldb, sb, dC, ldc, sc, m, n, k, batch, alpha, beta);
    };

    for (int i = 0; i < o.warmup; ++i) go();
    q.wait();

    // Wall clock over the whole run, one wait at the end.  Summing per-event
    // profiling intervals over queued submissions over-reports by ~6x here.
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < o.iters; ++i) go();
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)o.iters;

    // If the timed shape is cheap enough to reference on the CPU, do it: that
    // is a stronger statement than a check on a proxy shape.
    if (o.check && (double)m * n * k * batch <= 3.5e7) {
        std::vector<T> got(hC.size());
        // Recompute from the same start state rather than trying to unwind the
        // beta-scaled accumulation of warmup + iters passes.
        q.memcpy(dC, hC.data(), hC.size() * sizeof(T)).wait();
        go().wait();
        q.memcpy(got.data(), dC, hC.size() * sizeof(T)).wait();
        std::vector<T> ref;
        reference_gemm(m, n, k, batch, hA, lda, sa, hB, ldb, sb, hC, ldc, sc, alpha, beta,
                       ref);
        const double e = compare(got, ref);
        std::fprintf(stderr, "  check timed shape -> maxrelerr %.3e\n", e);
        maxrelerr = std::max(maxrelerr, e);
    }

    const double flops = (ST::is_complex ? 8.0 : 2.0) * (double)m * n * k * batch;
    const double tflops = flops / (ms * 1e-3) / 1e12;

    std::printf("RESULT dtype=%s m=%d n=%d k=%d batch=%d beta=%g tile=%s:%s ms=%.4f "
                "tflops=%.2f maxrelerr=%.3e\n",
                o.dtype.c_str(), m, n, k, batch, o.beta, cfg.c_str(),
                planar ? "planar" : "interleaved", ms, tflops, maxrelerr);

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);
    return 0;
}

int main(int argc, char** argv) {
    Opts o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_i = [&]() { return std::atoi(argv[++i]); };
        auto next_d = [&]() { return std::atof(argv[++i]); };
        auto next_s = [&]() { return std::string(argv[++i]); };
        if (a == "--m") o.m = next_i();
        else if (a == "--n") o.n = next_i();
        else if (a == "--k") o.k = next_i();
        else if (a == "--batch") o.batch = next_i();
        else if (a == "--iters") o.iters = next_i();
        else if (a == "--warmup") o.warmup = next_i();
        else if (a == "--beta") o.beta = next_d();
        else if (a == "--dtype") o.dtype = next_s();
        else if (a == "--tile") o.tile = next_s();
        else if (a == "--layout") o.layout = next_s();
        else if (a == "--device") o.device = next_s();
        else if (a == "--no-check") o.check = false;
        else if (a == "--list") {
            for (const auto& e : kCfgs) std::printf("%s\n", e.name);
            return 0;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--dtype float|double|cfloat|cdouble] [--m N] [--n N]"
                         " [--k N] [--batch N] [--beta X] [--tile NAME]"
                         " [--layout planar|interleaved] [--iters N] [--warmup N]"
                         " [--device gpu|cpu|any] [--no-check] [--list]\n",
                         argv[0]);
            return 2;
        }
    }
    if (o.dtype == "float") return run<float>(o);
    if (o.dtype == "double") return run<double>(o);
    if (o.dtype == "cfloat") return run<std::complex<float>>(o);
    if (o.dtype == "cdouble") return run<std::complex<double>>(o);
    std::fprintf(stderr, "unknown dtype '%s'\n", o.dtype.c_str());
    return 2;
}

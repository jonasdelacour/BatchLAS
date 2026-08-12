// ---------------------------------------------------------------------------
// tile-128x128-k8-t8x4 : a batched register-tiled GEMM for WIDE SCALARS.
//
// Standalone SYCL harness. Build:
//
//   /opt/dpcpp-cuda/bin/clang++ -O3 -std=c++20 -fsycl \
//       -fsycl-targets=nvidia_gpu_sm_89 --cuda-path=/usr/local/cuda-13.2 \
//       -Xcuda-ptxas -v tile-128x128-k8-t8x4.cpp -o tile-128x128-k8-t8x4
//
// ---------------------------------------------------------------------------
// WHY THIS SHAPE
//
// The float kernel (src/sycl/gemm/register_128x128.hh) uses a 128x128x8
// macro-tile with an 8x8 thread tile: 64 accumulators, 256 threads. That is
// 64 registers of accumulator for float, but 128 for double or
// std::complex<float> and 256 for std::complex<double> -- past the 255-register
// hardware limit, so it spills.
//
// This file keeps the macro-tile at 128x128x8 -- the shape that is known to
// work -- and halves only the thread tile, to 8 rows x 4 cols = 32
// accumulators, doubling the work-group to 16x32 = 512 threads. That is the
// most conservative change that fits: 32 accumulators is 64 registers for
// double and for complex<float>, the same accumulator budget the float kernel
// spent, on a scalar twice as wide.
//
// WHAT THAT COSTS, HONESTLY
//
// Per k-step a thread now consumes 8 A values and 4 B values (12 elements) to
// issue 32 multiply-accumulates, where the float kernel consumed 16 elements
// for 64. The element-level compute:load ratio drops from 4.0 to 2.67, a 1.5x
// regression, and it is real -- there is no way around it at half the tile.
//
// In *instructions* the drop looks worse still, because a 16-byte shared load
// no longer carries 4 elements. With VecLen = 16/sizeof(T):
//
//   float   V=4 : 8/4 + 4/4 = 3 LDS.128 per k-step for 64 FFMA  (8x8 tile: 4:64)
//   double  V=2 : 8/2 + 4/2 = 6 LDS.128 per k-step for 32 DFMA
//   cfloat  V=2 : 8/2 + 4/2 = 6 LDS.128 per k-step for 32 complex FMA
//   cdouble V=1 : 8/1 + 4/1 = 12 LDS.128 per k-step for 32 complex FMA
//
// So the nominal instruction ratio for double is 32:6 = 5.33:1, down from the
// float kernel's 16:1. Taken at face value that looks like a return to the
// 2.0-2.7:1 regime that saturated the shared pipe.
//
// It is not, and the reason is that the *unit* of work changed. The thing that
// has to be balanced is the FP pipe against the shared pipe, and both sides
// move when the scalar widens:
//
//   * double. An RTX 4090 is a consumer part: FP64 runs at 1/64 the FP32 rate,
//     2 DFMA/clk/SM against 128 FFMA/clk/SM. The shared pipe still delivers
//     128 B/clk/SM. 32 DFMA per k-step is 16 clocks of FP64 issue; the 6
//     LDS.128 that feed them are 6/4 = 1.5 clocks of shared issue per warp.
//     The shared pipe is over-provisioned by roughly an order of magnitude for
//     FP64 on this part. A 5.33:1 instruction ratio clears it with enormous
//     margin. For double the binding constraint is the DFMA rate, full stop,
//     and the honest target is cuBLAS DGEMM -- not a fraction of the FP32
//     number.
//
//   * complex<float>. One complex multiply-accumulate is 4 FFMA (written out
//     explicitly below as a*c-b*d, a*d+b*c). So 32 complex FMA per k-step is
//     128 FFMA, against 6 LDS.128 -- an effective 21.3:1 FFMA-per-shared-load
//     ratio, which is *better* than the float kernel's 16:1. Complex arithmetic
//     is denser per byte than real arithmetic, and that pays for the smaller
//     tile outright.
//
//   * complex<double> is the one scalar the 512-thread shape cannot carry. It
//     compiles to 226 registers, and 226 x 512 = 115,712 exceeds the 65,536-
//     entry register file, so the kernel would not launch at all. It is given
//     a 128x64x8 macro-tile instead (same 8x4 thread tile, 256 threads), which
//     lifts the per-thread budget to 256 and makes it fit. The cost is half the
//     B reuse; complex<double> really wants a 4x4 thread tile.
//
// MEASURED, sm_89, this file, -Xcuda-ptxas -v (all 16 instantiations spill 0
// bytes and use 0 bytes of stack frame). "regs" is fast-path / predicated:
//
//   dtype    tile          threads  regs      worst regs*threads  file    fits
//   float    128x128 8x4     512     66 / 63        33,792        65,536  yes
//   double   128x128 8x4     512    126 / 128       65,536        65,536  exact
//   cfloat   128x128 8x4     512    122 / 122       62,464        65,536  yes
//   cdouble  128x64  8x4     256    234 / 246       62,976        65,536  yes
//
// double's predicated kernel lands exactly on the register file: 128 x 512 =
// 65,536. It launches, at one block and 16 warps per SM -- the same 33%
// occupancy the float 128x128 kernel runs at -- but there is no headroom, so
// anything that adds a register to that kernel makes it unlaunchable.
//
// A TOOLCHAIN TRAP worth recording. The obvious way to give ptxas the block
// size is [[sycl::reqd_work_group_size(...)]]. Do not: on this DPC++ it lowers
// the three SYCL arguments straight into `.reqntid <arg0>, <arg1>` and drops
// arg2 entirely. The spec-correct order for a (1, 32, 16) range therefore emits
// `.reqntid 1, 32`, declaring a 32-thread block for a 512-thread launch --
// which both misleads ptxas about the register budget (it happily allocated
// 226 registers for complex<double>) and makes the launch itself invalid.
// Verified with a two-kernel probe: reqd_work_group_size(1,32,16) -> ".reqntid
// 1, 32", reqd_work_group_size(16,32,1) -> ".reqntid 16, 32". The attribute is
// omitted below and the register numbers above are what ptxas picks unaided.
//
// PTX evidence that the layout rules actually took (per fast-path kernel):
//
//   dtype    shared loads             inner-loop FMA
//   float    24 x ld.shared.v4.b32    288 fma.rn.f32   (8 k-steps x 3 loads)
//   double   48 x ld.shared.v2.b64    288 fma.rn.f64   (8 k-steps x 6 loads)
//   cfloat   48 x ld.shared.v4.b32   1280 fma.rn.f32   (8 k-steps x 6 loads)
//   cdouble  96 x ld.shared.v2.b64   1280 fma.rn.f64   (8 k-steps x 12 loads)
//
// Every shared load is 16 bytes; the scalar ld.shared count is 0 in all four,
// which is the failure mode note 2 exists to prevent. No __mulsc3 / __muldc3
// call appears in any kernel, which is what note 6 exists to prevent.
//
// WHAT IS PRESERVED FROM THE FLOAT KERNEL
//
//   1. Vectorized 16-byte shared loads. VecLen is derived from sizeof(T) so a
//      fragment load is always exactly one LDS.128 regardless of scalar.
//   2. Shared strides are exactly TileM / TileN, never TileM+1. An odd stride
//      means the compiler cannot prove 16-byte alignment and every fragment
//      load degrades to scalar ld.shared.
//   3. B is staged [k][n], not [n][k], so a thread's B values are contiguous
//      and vectorize.
//   4. The band split. The thread's TTM rows are split into TTM/VecLen bands
//      spread across the 128-wide tile at stride TileM/(TTM/VecLen), so an
//      LDS.128 -- serviced 8 lanes at a time, 8 lanes x 16 B = exactly the 32
//      banks -- is bank-conflict free. For double this is 4 bands of 2 at
//      stride 32 rather than the float kernel's 2 bands of 4 at stride 64.
//   5. The m index is the fastest-varying thread index (local_id(2)), so the
//      epilogue's read and write of a column-major C are coalesced. Getting
//      this backwards is free at beta == 0 and catastrophic at beta != 0; this
//      harness therefore drives beta from argv and verifies at beta != 0.
//   6. Complex multiplies are written out by hand. std::complex operator*
//      emits an isnan branch and a call to __mulsc3 in device code.
//   7. Accumulators are a plain local array in the kernel body. An
//      out-parameter passed by reference has been measured to spill them.
// ---------------------------------------------------------------------------

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
#include <vector>

// ===========================================================================
// Scalar plumbing.
//
// T is the *memory* type (double, std::complex<float>, ...). The inner loop
// never touches std::complex: it converts to a plain POD Cplx<R> on load, so
// that no operator of libstdc++'s complex can sneak an isnan branch or a call
// to __mulsc3 into the hot loop.
// ===========================================================================

template <typename R>
struct Cplx {
    R re;
    R im;
};

template <typename T>
struct Scalar {
    using compute = T;
    using real = T;
    static constexpr bool is_complex = false;
    static compute load(const T& x) { return x; }
    static T store(const compute& x) { return x; }
    static compute zero() { return compute(0); }
    static compute make(double re, double) { return compute(re); }
};

template <typename R>
struct Scalar<std::complex<R>> {
    using compute = Cplx<R>;
    using real = R;
    static constexpr bool is_complex = true;
    static compute load(const std::complex<R>& x) { return compute{x.real(), x.imag()}; }
    static std::complex<R> store(const compute& x) { return std::complex<R>(x.re, x.im); }
    static compute zero() { return compute{R(0), R(0)}; }
    static compute make(double re, double im) { return compute{R(re), R(im)}; }
};

// --- multiply-accumulate -----------------------------------------------------
// Real: one statement, so -ffp-contract folds it to a single FFMA/DFMA.
template <typename C>
inline void mac(C& acc, const C& a, const C& b) {
    acc += a * b;
}

// Complex: written out as (a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re).
// Four statements, four FFMA/DFMA, no isnan branch, no __mulsc3.
template <typename R>
inline void mac(Cplx<R>& acc, const Cplx<R>& a, const Cplx<R>& b) {
    acc.re += a.re * b.re;
    acc.re -= a.im * b.im;
    acc.im += a.re * b.im;
    acc.im += a.im * b.re;
}

// --- epilogue arithmetic -----------------------------------------------------
template <typename C>
inline C cmul(const C& a, const C& b) {
    return a * b;
}

template <typename R>
inline Cplx<R> cmul(const Cplx<R>& a, const Cplx<R>& b) {
    return Cplx<R>{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

template <typename C>
inline C axpby(const C& alpha, const C& x, const C& beta, const C& y) {
    return alpha * x + beta * y;
}

template <typename R>
inline Cplx<R> axpby(const Cplx<R>& alpha, const Cplx<R>& x, const Cplx<R>& beta,
                     const Cplx<R>& y) {
    Cplx<R> r;
    r.re = alpha.re * x.re - alpha.im * x.im + beta.re * y.re - beta.im * y.im;
    r.im = alpha.re * x.im + alpha.im * x.re + beta.re * y.im + beta.im * y.re;
    return r;
}

template <typename C>
inline bool is_zero(const C& x) {
    return x == C(0);
}

template <typename R>
inline bool is_zero(const Cplx<R>& x) {
    return x.re == R(0) && x.im == R(0);
}

// --- the 16-byte packet ------------------------------------------------------
// A plain struct rather than sycl::vec, so the generated access is the same as
// the equivalent CUDA float4/double2 access. alignas is what lets the compiler
// prove the load is one LDS.128 / STS.128.
template <typename T, int N>
struct alignas(N * sizeof(T)) VecN {
    T v[N];
};

// ===========================================================================
// Tile configuration.
// ===========================================================================

template <typename T, int TileM_, int TileN_, int TileK_, int TTM_, int TTN_>
struct TileCfg {
    static constexpr int TileM = TileM_;
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;
    static constexpr int TTM = TTM_;  // thread tile rows (m)
    static constexpr int TTN = TTN_;  // thread tile cols (n)

    // 16 bytes is one LDS.128 / STS.128 / one sector of a coalesced global
    // access. Never let the "vector" exceed the thread tile in either
    // direction, or the band split stops making sense.
    static constexpr int RawVec = static_cast<int>(16 / sizeof(T));
    static constexpr int VecLen = RawVec < 1 ? 1 : (RawVec > TTN ? TTN : RawVec);

    static constexpr int LocalRows = TileM / TTM;          // the m direction
    static constexpr int LocalCols = TileN / TTN;          // the n direction
    static constexpr int Threads = LocalRows * LocalCols;

    static_assert(Threads <= 1024, "work group too large");
    // Registers per thread are capped by the 64K register file:
    // Threads * regs <= 65536, so a launch needs regs <= 65536/Threads.
    static constexpr int RegBudget = 65536 / Threads;
};

template <typename T>
struct WideTile : TileCfg<T, 128, 128, 8, 8, 4> {};

// std::complex<double> is the one scalar the 128x128 / 512-thread shape cannot
// carry. 32 accumulators is 128 registers on their own, and measurement puts
// the whole kernel at 226; at 512 threads the register file allows only 128 per
// thread, so the kernel would not even launch. Halving the macro-tile in n
// halves the work group to 256 threads, which lifts the budget to 256 registers
// and makes it fit -- at the cost of half the B reuse. This is recorded here
// rather than hidden: complex<double> wants a 4x4 thread tile, not this one.
template <>
struct WideTile<std::complex<double>> : TileCfg<std::complex<double>, 128, 64, 8, 8, 4> {};

template <typename T>
inline std::string tile_label() {
    using C = WideTile<T>;
    return std::to_string(C::TileM) + "x" + std::to_string(C::TileN) + "x" +
           std::to_string(C::TileK) + "_t" + std::to_string(C::TTM) + "x" +
           std::to_string(C::TTN);
}

// ===========================================================================
// The kernel.
// ===========================================================================

template <typename T, bool Fast>
class WideGemmKernel;

// A and B are column-major, NoTrans:
//    A is m x k, element (i,kk) at a[b*stride_a + kk*lda + i]
//    B is k x n, element (kk,j) at b[b*stride_b + j*ldb + kk]
//    C is m x n, element (i,j)  at c[b*stride_c + j*ldc + i]
//
// Fast == true assumes m%TileM==0, n%TileN==0, k%TileK==0 and VecLen-aligned
// pointers / ld / stride, and is completely unpredicated.
template <typename T, bool Fast>
sycl::event launch_wide_gemm(sycl::queue& q, int m, int n, int k, int batch,
                             const T* a_ptr, int lda, long long stride_a,
                             const T* b_ptr, int ldb, long long stride_b,
                             T* c_ptr, int ldc, long long stride_c,
                             typename Scalar<T>::compute alpha,
                             typename Scalar<T>::compute beta) {
    using Cfg = WideTile<T>;
    constexpr int TileM = Cfg::TileM;
    constexpr int TileN = Cfg::TileN;
    constexpr int TileK = Cfg::TileK;
    constexpr int TTM = Cfg::TTM;
    constexpr int TTN = Cfg::TTN;
    constexpr int V = Cfg::VecLen;
    constexpr int LocalRows = Cfg::LocalRows;
    constexpr int LocalCols = Cfg::LocalCols;
    constexpr int Threads = Cfg::Threads;

    // The band split. TTM rows become NBM bands of V, placed SecM apart so the
    // bands span the whole 128-wide tile; likewise for n.
    constexpr int NBM = TTM / V;
    constexpr int SecM = TileM / NBM;
    constexpr int NBN = TTN / V;
    constexpr int SecN = TileN / NBN;

    // No padding. An aligned stride is what lets the fragment loads vectorize.
    constexpr int AStride = TileM;  // A staged [k][m]
    constexpr int BStride = TileN;  // B staged [k][n]

    // Staging: how many elements each thread moves per tile, and how wide a
    // packet it can use to move them.
    constexpr int APerT = TileM * TileK / Threads;  // 2
    constexpr int AV = APerT < V ? APerT : V;
    constexpr int APkts = APerT / AV;
    constexpr int ASlots = TileM / AV;  // packet slots along m
    constexpr int BPerT = TileK * TileN / Threads;  // 2
    constexpr int BV = BPerT < V ? BPerT : V;
    constexpr int BPkts = BPerT / BV;
    constexpr int BSlots = TileK / BV;  // packet slots along k within a column

    static_assert(TTM % V == 0 && TTN % V == 0,
                  "thread tile must be a whole number of vector bands");
    static_assert(LocalRows * V == SecM, "m band split must tile the macro-tile exactly");
    static_assert(LocalCols * V == SecN, "n band split must tile the macro-tile exactly");
    static_assert(APerT % AV == 0 && BPerT % BV == 0, "staging must divide evenly");
    static_assert(TileM * TileK == Threads * APerT, "A staging must cover the tile exactly");
    static_assert(TileK * TileN == Threads * BPerT, "B staging must cover the tile exactly");
    static_assert(AStride % V == 0 && BStride % V == 0,
                  "shared stride must preserve packet alignment");

    using CT = typename Scalar<T>::compute;
    using FragVec = VecN<T, V>;
    using AVec = VecN<T, AV>;
    using BVec = VecN<T, BV>;

    const int group_rows = (m + TileM - 1) / TileM;
    const int group_cols = (n + TileN - 1) / TileN;

    // Dimension 2 is the fastest-varying one, and it carries m. See note 5 in
    // the header: this is what keeps the epilogue's C traffic coalesced.
    const sycl::range<3> local(1, LocalCols, LocalRows);
    const sycl::range<3> global(static_cast<size_t>(batch),
                                static_cast<size_t>(group_cols) * LocalCols,
                                static_cast<size_t>(group_rows) * LocalRows);

    return q.submit([&](sycl::handler& h) {
        // Allocated as packets so the base is provably 16-byte aligned; the
        // kernel then views them as T*.
        sycl::local_accessor<FragVec, 1> tile_a(sycl::range<1>(TileK * AStride / V), h);
        sycl::local_accessor<FragVec, 1> tile_b(sycl::range<1>(TileK * BStride / V), h);

        // NOTE: do NOT put [[sycl::reqd_work_group_size(1, LocalCols, LocalRows)]]
        // on this lambda. On this toolchain it lowers to `.reqntid 1, 32` --
        // the SYCL dim0/dim1 pair emitted straight through as CUDA ntid.x/ntid.y
        // with the dim2 extent dropped -- which declares a 32-thread block for a
        // 512-thread launch. ptxas then sizes registers against 32 threads (it
        // allowed 226 for complex<double>) and the launch itself is rejected.
        // Without the attribute every instantiation is launchable on its own
        // merits; see the register budget printed by run().
        h.parallel_for<WideGemmKernel<T, Fast>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));

                const int ty = static_cast<int>(item.get_local_id(2));  // 0..15, m
                const int tx = static_cast<int>(item.get_local_id(1));  // 0..31, n
                const int tid = tx * LocalRows + ty;                    // linear local id

                const int m0 = static_cast<int>(item.get_group(2)) * TileM;
                const int n0 = static_cast<int>(item.get_group(1)) * TileN;

                const T* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                const T* Bb = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
                T* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;

                T* sa = reinterpret_cast<T*>(
                    tile_a.template get_multi_ptr<sycl::access::decorated::no>().get());
                T* sb = reinterpret_cast<T*>(
                    tile_b.template get_multi_ptr<sycl::access::decorated::no>().get());

                // Plain local array. Never an out-parameter by reference.
                CT accum[TTM][TTN];
#pragma unroll
                for (int i = 0; i < TTM; ++i) {
#pragma unroll
                    for (int j = 0; j < TTN; ++j) {
                        accum[i][j] = Scalar<T>::zero();
                    }
                }

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    // ---- global -> shared ----------------------------------
                    if constexpr (Fast) {
                        // A: lanes walk down m, the contiguous direction of a
                        // column-major A, so the warp is fully coalesced.
#pragma unroll
                        for (int p = 0; p < APkts; ++p) {
                            const int slot = tid + p * Threads;
                            const int am = (slot % ASlots) * AV;
                            const int ak = slot / ASlots;
                            *reinterpret_cast<AVec*>(&sa[ak * AStride + am]) =
                                *reinterpret_cast<const AVec*>(
                                    Ab + (m0 + am) +
                                    static_cast<std::ptrdiff_t>(k0 + ak) * lda);
                        }
                        // B: read down k (contiguous in a column-major B) and
                        // transpose into shared so that n ends up contiguous.
                        // The shared side of the transpose cannot vectorize.
#pragma unroll
                        for (int p = 0; p < BPkts; ++p) {
                            const int slot = tid + p * Threads;
                            const int bk = (slot % BSlots) * BV;
                            const int bn = slot / BSlots;
                            const BVec vb = *reinterpret_cast<const BVec*>(
                                Bb + (k0 + bk) + static_cast<std::ptrdiff_t>(n0 + bn) * ldb);
#pragma unroll
                            for (int i = 0; i < BV; ++i) {
                                sb[(bk + i) * BStride + bn] = vb.v[i];
                            }
                        }
                    } else {
                        // Predicated staging. The shared tile is always filled
                        // to its full TileM x TileK with zeros outside the
                        // matrix, so the inner loop needs no bounds checks.
#pragma unroll
                        for (int p = 0; p < APkts; ++p) {
                            const int slot = tid + p * Threads;
                            const int am = (slot % ASlots) * AV;
                            const int ak = slot / ASlots;
                            const int gk = k0 + ak;
#pragma unroll
                            for (int i = 0; i < AV; ++i) {
                                const int gm = m0 + am + i;
                                sa[ak * AStride + am + i] =
                                    (gm < m && gk < k)
                                        ? Ab[gm + static_cast<std::ptrdiff_t>(gk) * lda]
                                        : T(0);
                            }
                        }
#pragma unroll
                        for (int p = 0; p < BPkts; ++p) {
                            const int slot = tid + p * Threads;
                            const int bk = (slot % BSlots) * BV;
                            const int bn = slot / BSlots;
                            const int gn = n0 + bn;
#pragma unroll
                            for (int i = 0; i < BV; ++i) {
                                const int gk = k0 + bk + i;
                                sb[(bk + i) * BStride + bn] =
                                    (gk < k && gn < n)
                                        ? Bb[gk + static_cast<std::ptrdiff_t>(gn) * ldb]
                                        : T(0);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // ---- shared -> registers -> FMA ------------------------
                    // NBM + NBN vector loads feed TTM*TTN multiply-accumulates.
#pragma unroll
                    for (int kk = 0; kk < TileK; ++kk) {
                        CT af[TTM];
                        CT bf[TTN];
#pragma unroll
                        for (int band = 0; band < NBM; ++band) {
                            const FragVec fa = *reinterpret_cast<const FragVec*>(
                                &sa[kk * AStride + band * SecM + ty * V]);
#pragma unroll
                            for (int i = 0; i < V; ++i) {
                                af[band * V + i] = Scalar<T>::load(fa.v[i]);
                            }
                        }
#pragma unroll
                        for (int band = 0; band < NBN; ++band) {
                            const FragVec fb = *reinterpret_cast<const FragVec*>(
                                &sb[kk * BStride + band * SecN + tx * V]);
#pragma unroll
                            for (int j = 0; j < V; ++j) {
                                bf[band * V + j] = Scalar<T>::load(fb.v[j]);
                            }
                        }
#pragma unroll
                        for (int i = 0; i < TTM; ++i) {
#pragma unroll
                            for (int j = 0; j < TTN; ++j) {
                                mac(accum[i][j], af[i], bf[j]);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // ---- epilogue ------------------------------------------------
                // Within a band the V rows are consecutive in m, which is the
                // contiguous direction of a column-major C, so a band is one
                // 16-byte access. Adjacent lanes differ in ty, hence in m, so
                // 8 lanes cover 128 contiguous bytes -- both for the store and
                // for the beta != 0 read.
                const bool beta_is_zero = is_zero(beta);
#pragma unroll
                for (int band = 0; band < NBM; ++band) {
                    const int gm = m0 + band * SecM + ty * V;
#pragma unroll
                    for (int j = 0; j < TTN; ++j) {
                        const int nb = j / V;
                        const int jj = j % V;
                        const int gn = n0 + nb * SecN + tx * V + jj;
                        if constexpr (Fast) {
                            T* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                            FragVec out;
                            if (beta_is_zero) {
#pragma unroll
                                for (int i = 0; i < V; ++i) {
                                    out.v[i] = Scalar<T>::store(
                                        cmul(alpha, accum[band * V + i][j]));
                                }
                            } else {
                                const FragVec prior =
                                    *reinterpret_cast<const FragVec*>(p);
#pragma unroll
                                for (int i = 0; i < V; ++i) {
                                    out.v[i] = Scalar<T>::store(
                                        axpby(alpha, accum[band * V + i][j], beta,
                                              Scalar<T>::load(prior.v[i])));
                                }
                            }
                            *reinterpret_cast<FragVec*>(p) = out;
                        } else {
                            if (gn >= n) {
                                continue;
                            }
#pragma unroll
                            for (int i = 0; i < V; ++i) {
                                const int row = gm + i;
                                if (row >= m) {
                                    continue;
                                }
                                T* p = &Cb[row + static_cast<std::ptrdiff_t>(gn) * ldc];
                                const CT acc = accum[band * V + i][j];
                                *p = Scalar<T>::store(
                                    beta_is_zero
                                        ? cmul(alpha, acc)
                                        : axpby(alpha, acc, beta, Scalar<T>::load(*p)));
                            }
                        }
                    }
                }
            });
    });
}

// Does this problem satisfy everything the unpredicated path assumes?
template <typename T>
bool can_use_fast_path(int m, int n, int k, const T* a, int lda, long long stride_a,
                       const T* b, int ldb, long long stride_b, const T* c, int ldc,
                       long long stride_c) {
    using Cfg = WideTile<T>;
    constexpr int V = Cfg::VecLen;
    if (m % Cfg::TileM || n % Cfg::TileN || k % Cfg::TileK) {
        return false;
    }
    auto ok = [](const T* p, int ld, long long stride) {
        return p != nullptr &&
               (reinterpret_cast<std::uintptr_t>(p) % 16u) == 0 && (ld % V) == 0 &&
               (stride % V) == 0;
    };
    return ok(a, lda, stride_a) && ok(b, ldb, stride_b) && ok(c, ldc, stride_c);
}

template <typename T>
sycl::event launch_wide_gemm_auto(sycl::queue& q, int m, int n, int k, int batch,
                                  const T* a, int lda, long long stride_a, const T* b,
                                  int ldb, long long stride_b, T* c, int ldc,
                                  long long stride_c, typename Scalar<T>::compute alpha,
                                  typename Scalar<T>::compute beta, bool* used_fast) {
    const bool fast =
        can_use_fast_path<T>(m, n, k, a, lda, stride_a, b, ldb, stride_b, c, ldc, stride_c);
    if (used_fast) {
        *used_fast = fast;
    }
    if (fast) {
        return launch_wide_gemm<T, true>(q, m, n, k, batch, a, lda, stride_a, b, ldb,
                                         stride_b, c, ldc, stride_c, alpha, beta);
    }
    return launch_wide_gemm<T, false>(q, m, n, k, batch, a, lda, stride_a, b, ldb,
                                      stride_b, c, ldc, stride_c, alpha, beta);
}

// ===========================================================================
// Host side: naming, widening, reference, verification, timing.
// ===========================================================================

template <typename T>
struct HostT;

template <>
struct HostT<float> {
    using wide = double;
    static constexpr const char* name = "float";
    static constexpr int flops_per_mac = 2;
    static float make(double re, double) { return static_cast<float>(re); }
    static wide up(const float& x) { return static_cast<double>(x); }
    static double mag(const wide& x) { return std::fabs(x); }
};

template <>
struct HostT<double> {
    using wide = double;
    static constexpr const char* name = "double";
    static constexpr int flops_per_mac = 2;
    static double make(double re, double) { return re; }
    static wide up(const double& x) { return x; }
    static double mag(const wide& x) { return std::fabs(x); }
};

template <>
struct HostT<std::complex<float>> {
    using wide = std::complex<double>;
    static constexpr const char* name = "cfloat";
    static constexpr int flops_per_mac = 8;
    static std::complex<float> make(double re, double im) {
        return std::complex<float>(static_cast<float>(re), static_cast<float>(im));
    }
    static wide up(const std::complex<float>& x) {
        return wide(static_cast<double>(x.real()), static_cast<double>(x.imag()));
    }
    static double mag(const wide& x) { return std::abs(x); }
};

template <>
struct HostT<std::complex<double>> {
    using wide = std::complex<double>;
    static constexpr const char* name = "cdouble";
    static constexpr int flops_per_mac = 8;
    static std::complex<double> make(double re, double im) {
        return std::complex<double>(re, im);
    }
    static wide up(const std::complex<double>& x) { return x; }
    static double mag(const wide& x) { return std::abs(x); }
};

// Deterministic, reproducible fill in roughly [-0.5, 0.5].
static inline double det_val(std::uint64_t i, std::uint64_t salt) {
    std::uint64_t h = i * 6364136223846793005ull + salt * 1442695040888963407ull + 1ull;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 29;
    return static_cast<double>(h % 2000u) / 2000.0 - 0.5;
}

template <typename T>
void fill(std::vector<T>& v, std::uint64_t salt) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = HostT<T>::make(det_val(i, salt), det_val(i, salt + 977u));
    }
}

// Column-major reference, accumulated in the widened type.
template <typename T>
void reference_gemm(int m, int n, int k, int batch, const std::vector<T>& A, int lda,
                    long long sa, const std::vector<T>& B, int ldb, long long sb,
                    std::vector<T>& C, int ldc, long long sc,
                    typename HostT<T>::wide alpha, typename HostT<T>::wide beta) {
    using W = typename HostT<T>::wide;
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                W acc = W(0);
                for (int kk = 0; kk < k; ++kk) {
                    acc += HostT<T>::up(A[b * sa + static_cast<long long>(kk) * lda + i]) *
                           HostT<T>::up(B[b * sb + static_cast<long long>(j) * ldb + kk]);
                }
                T& dst = C[b * sc + static_cast<long long>(j) * ldc + i];
                const W prior = HostT<T>::up(dst);
                const W out = alpha * acc + beta * prior;
                dst = HostT<T>::make(
                    [](const W& w) {
                        if constexpr (std::is_same_v<W, std::complex<double>>) {
                            return w.real();
                        } else {
                            return w;
                        }
                    }(out),
                    [](const W& w) {
                        if constexpr (std::is_same_v<W, std::complex<double>>) {
                            return w.imag();
                        } else {
                            return 0.0;
                        }
                    }(out));
            }
        }
    }
}

template <typename T>
typename Scalar<T>::compute to_compute(double re, double im) {
    return Scalar<T>::make(re, im);
}

template <typename T>
typename HostT<T>::wide to_wide(double re, double im) {
    using W = typename HostT<T>::wide;
    if constexpr (std::is_same_v<W, std::complex<double>>) {
        return W(re, im);
    } else {
        (void)im;
        return W(re);
    }
}

struct CaseSpec {
    int m, n, k, batch;
    double beta_re;
    double beta_im;
    const char* label;
};

// Runs one shape on the device, compares against the host reference, returns
// the normwise max relative error.
template <typename T>
double verify_case(sycl::queue& q, const CaseSpec& cs, double alpha_re, double alpha_im,
                   bool verbose) {
    const int m = cs.m, n = cs.n, k = cs.k, batch = cs.batch;
    const int lda = m, ldb = k, ldc = m;
    const long long sa = static_cast<long long>(m) * k;
    const long long sb = static_cast<long long>(k) * n;
    const long long sc = static_cast<long long>(m) * n;

    std::vector<T> hA(static_cast<std::size_t>(sa) * batch);
    std::vector<T> hB(static_cast<std::size_t>(sb) * batch);
    std::vector<T> hC(static_cast<std::size_t>(sc) * batch);
    fill(hA, 11u);
    fill(hB, 29u);
    fill(hC, 53u);

    std::vector<T> ref = hC;
    reference_gemm<T>(m, n, k, batch, hA, lda, sa, hB, ldb, sb, ref, ldc, sc,
                      to_wide<T>(alpha_re, alpha_im), to_wide<T>(cs.beta_re, cs.beta_im));

    T* dA = sycl::malloc_device<T>(hA.size(), q);
    T* dB = sycl::malloc_device<T>(hB.size(), q);
    T* dC = sycl::malloc_device<T>(hC.size(), q);
    q.memcpy(dA, hA.data(), hA.size() * sizeof(T)).wait();
    q.memcpy(dB, hB.data(), hB.size() * sizeof(T)).wait();
    q.memcpy(dC, hC.data(), hC.size() * sizeof(T)).wait();

    bool fast = false;
    launch_wide_gemm_auto<T>(q, m, n, k, batch, dA, lda, sa, dB, ldb, sb, dC, ldc, sc,
                             to_compute<T>(alpha_re, alpha_im),
                             to_compute<T>(cs.beta_re, cs.beta_im), &fast)
        .wait();

    std::vector<T> got(hC.size());
    q.memcpy(got.data(), dC, got.size() * sizeof(T)).wait();
    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);

    double max_abs_err = 0.0;
    double max_ref = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const auto r = HostT<T>::up(ref[i]);
        const auto g = HostT<T>::up(got[i]);
        max_ref = std::max(max_ref, HostT<T>::mag(r));
        max_abs_err = std::max(max_abs_err, HostT<T>::mag(g - r));
    }
    const double rel = max_ref > 0.0 ? max_abs_err / max_ref : max_abs_err;
    if (verbose) {
        std::fprintf(stderr,
                     "  check %-10s %4dx%4dx%4d batch %d beta=(%g,%g) path=%s  relerr=%.3e\n",
                     cs.label, m, n, k, batch, cs.beta_re, cs.beta_im,
                     fast ? "fast" : "predicated", rel);
    }
    return rel;
}

template <typename T>
int run(int m, int n, int k, int batch, double beta_re, int iters, int warmup,
        bool skip_timing) {
    // In-order, deliberately. A default out-of-order queue would let the timed
    // iterations overlap: they all read-modify-write the same C, so at beta != 0
    // that is a genuine data race, and the wall clock would then report the
    // throughput of several overlapping GEMMs rather than the latency of one.
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    std::fprintf(stderr, "device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());

    using Cfg = WideTile<T>;
    const int shared_bytes =
        static_cast<int>(Cfg::TileK * (Cfg::TileM + Cfg::TileN) * sizeof(T));
    const int nbm = Cfg::TTM / Cfg::VecLen;
    const int nbn = Cfg::TTN / Cfg::VecLen;
    const std::string label = tile_label<T>();
    std::fprintf(stderr,
                 "tile %s: macro %dx%dx%d, thread tile %dx%d (%d accumulators), "
                 "%d threads, VecLen=%d, shared=%d B, register budget=%d/thread\n"
                 "  per k-step: %d vector shared loads : %d MACs "
                 "(= %d FFMA-equivalent, ratio %.2f:1)\n",
                 label.c_str(), Cfg::TileM, Cfg::TileN, Cfg::TileK, Cfg::TTM, Cfg::TTN,
                 Cfg::TTM * Cfg::TTN, Cfg::Threads, Cfg::VecLen, shared_bytes,
                 Cfg::RegBudget, nbm + nbn, Cfg::TTM * Cfg::TTN,
                 Cfg::TTM * Cfg::TTN * (Scalar<T>::is_complex ? 4 : 1),
                 double(Cfg::TTM * Cfg::TTN * (Scalar<T>::is_complex ? 4 : 1)) /
                     double(nbm + nbn));

    // Non-trivial alpha so a missing scale cannot pass, and a complex beta so
    // the read-modify-write path is exercised in both components.
    const double alpha_re = 1.25;
    const double alpha_im = Scalar<T>::is_complex ? -0.5 : 0.0;
    const double beta_im = Scalar<T>::is_complex ? 0.5 * beta_re : 0.0;

    // --- correctness, always, before anything is timed ---------------------
    std::fprintf(stderr, "verifying...\n");
    double maxrelerr = 0.0;
    const CaseSpec cases[] = {
        // Aligned fast path, beta == 0 and beta != 0.
        {128, 128, 32, 3, 0.0, 0.0, "fast/b0"},
        {128, 128, 32, 3, 0.75, Scalar<T>::is_complex ? 0.375 : 0.0, "fast/b1"},
        {256, 128, 64, 2, beta_re, beta_im, "fast/arg"},
        // Ragged: exercises the predicated path and the tail logic.
        {100, 70, 13, 2, 0.75, Scalar<T>::is_complex ? 0.375 : 0.0, "ragged"},
        {129, 257, 9, 1, 0.0, 0.0, "ragged/b0"},
    };
    for (const auto& cs : cases) {
        maxrelerr = std::max(maxrelerr, verify_case<T>(q, cs, alpha_re, alpha_im, true));
    }

    const double tol = (sizeof(typename Scalar<T>::real) == 4) ? 5e-5 : 1e-12;
    if (!(maxrelerr <= tol)) {
        std::fprintf(stderr, "FAIL: maxrelerr %.3e exceeds tolerance %.3e\n", maxrelerr, tol);
        std::printf("RESULT dtype=%s m=%d n=%d k=%d batch=%d beta=%g "
                    "tile=%s ms=nan tflops=nan maxrelerr=%.3e\n",
                    HostT<T>::name, m, n, k, batch, beta_re, label.c_str(), maxrelerr);
        return 1;
    }
    std::fprintf(stderr, "verification OK (maxrelerr %.3e <= %.3e)\n", maxrelerr, tol);

    if (skip_timing) {
        std::printf("RESULT dtype=%s m=%d n=%d k=%d batch=%d beta=%g "
                    "tile=%s ms=nan tflops=nan maxrelerr=%.3e\n",
                    HostT<T>::name, m, n, k, batch, beta_re, label.c_str(), maxrelerr);
        return 0;
    }

    // --- timed sweep --------------------------------------------------------
    const int lda = m, ldb = k, ldc = m;
    const long long sa = static_cast<long long>(m) * k;
    const long long sb = static_cast<long long>(k) * n;
    const long long sc = static_cast<long long>(m) * n;
    const std::size_t nA = static_cast<std::size_t>(sa) * batch;
    const std::size_t nB = static_cast<std::size_t>(sb) * batch;
    const std::size_t nC = static_cast<std::size_t>(sc) * batch;

    T* dA = sycl::malloc_device<T>(nA, q);
    T* dB = sycl::malloc_device<T>(nB, q);
    T* dC = sycl::malloc_device<T>(nC, q);
    if (!dA || !dB || !dC) {
        std::fprintf(stderr, "allocation failed (%.2f GB requested)\n",
                     double((nA + nB + nC) * sizeof(T)) / 1e9);
        return 2;
    }
    {
        std::vector<T> hA(nA);
        fill(hA, 11u);
        q.memcpy(dA, hA.data(), nA * sizeof(T)).wait();
    }
    {
        std::vector<T> hB(nB);
        fill(hB, 29u);
        q.memcpy(dB, hB.data(), nB * sizeof(T)).wait();
    }
    {
        // C must be initialised: at beta != 0 the kernel reads it, and reading
        // uninitialised device memory would put NaNs in the pipeline.
        std::vector<T> hC(nC);
        fill(hC, 53u);
        q.memcpy(dC, hC.data(), nC * sizeof(T)).wait();
    }

    const auto alpha = to_compute<T>(alpha_re, alpha_im);
    const auto beta = to_compute<T>(beta_re, beta_im);
    // Decided up front rather than read back out of the launcher, so the label
    // is right even when --warmup 0 means the lambda has not run yet.
    bool fast = can_use_fast_path<T>(m, n, k, dA, lda, sa, dB, ldb, sb, dC, ldc, sc);
    std::fprintf(stderr, "timed path: %s\n", fast ? "fast (unpredicated)" : "predicated");
    auto once = [&]() {
        return launch_wide_gemm_auto<T>(q, m, n, k, batch, dA, lda, sa, dB, ldb, sb, dC,
                                        ldc, sc, alpha, beta, &fast);
    };

    for (int i = 0; i < warmup; ++i) {
        once();
    }
    q.wait();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        once();
    }
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double flop = double(HostT<T>::flops_per_mac) * double(m) * double(n) *
                        double(k) * double(batch);
    const double tflops = flop / (ms * 1e-3) / 1e12;

    sycl::free(dA, q);
    sycl::free(dB, q);
    sycl::free(dC, q);

    std::printf("RESULT dtype=%s m=%d n=%d k=%d batch=%d beta=%g tile=%s "
                "ms=%.4f tflops=%.3f maxrelerr=%.3e\n",
                HostT<T>::name, m, n, k, batch, beta_re, label.c_str(), ms, tflops,
                maxrelerr);
    return 0;
}

int main(int argc, char** argv) {
    int m = 512, n = 512, k = 512, batch = 64;
    int iters = 20, warmup = 5;
    double beta = 1.0;  // beta != 0 by default: a beta = 0 harness cannot see
                        // the epilogue-orientation defect.
    std::string dtype = "double";
    bool skip_timing = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next_int = [&]() { return std::atoi(argv[++i]); };
        auto next_str = [&]() { return std::string(argv[++i]); };
        if (a == "--m") m = next_int();
        else if (a == "--n") n = next_int();
        else if (a == "--k") k = next_int();
        else if (a == "--batch") batch = next_int();
        else if (a == "--iters") iters = next_int();
        else if (a == "--warmup") warmup = next_int();
        else if (a == "--beta") beta = std::atof(argv[++i]);
        else if (a == "--dtype") dtype = next_str();
        else if (a == "--check-only") skip_timing = true;
        else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                         "usage: %s [--m N] [--n N] [--k N] [--batch N] "
                         "[--dtype float|double|cfloat|cdouble] [--beta X] "
                         "[--iters N] [--warmup N] [--check-only]\n",
                         argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }

    if (m <= 0 || n <= 0 || k <= 0 || batch <= 0) {
        std::fprintf(stderr, "m, n, k, batch must all be positive\n");
        return 2;
    }

    try {
        if (dtype == "double" || dtype == "f64") {
            return run<double>(m, n, k, batch, beta, iters, warmup, skip_timing);
        }
        if (dtype == "float" || dtype == "f32") {
            return run<float>(m, n, k, batch, beta, iters, warmup, skip_timing);
        }
        if (dtype == "cfloat" || dtype == "complex64" || dtype == "c64") {
            return run<std::complex<float>>(m, n, k, batch, beta, iters, warmup,
                                            skip_timing);
        }
        if (dtype == "cdouble" || dtype == "complex128" || dtype == "c128") {
            return run<std::complex<double>>(m, n, k, batch, beta, iters, warmup,
                                             skip_timing);
        }
    } catch (const sycl::exception& e) {
        // A launch that asks for more registers than the SM can supply lands
        // here, as does an out-of-memory allocation. Say so plainly rather than
        // dying with an unhandled exception.
        std::fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 3;
    }
    std::fprintf(stderr, "unknown dtype '%s'\n", dtype.c_str());
    return 2;
}

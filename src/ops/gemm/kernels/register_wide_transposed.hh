#pragma once

// A register-tiled GEMM for TRANSPOSED and CONJUGATE-TRANSPOSED operands with
// a wide scalar (double, complex<float>, complex<double>).
//
// register_tiled_common.hh goes through Annex-G std::complex operator*, and
// register_64x64_k16_wide.hh is NN only. The macro tile is a PARAMETER matched
// to a panel width (16 or 32), not square.
// evidence: docs/perf/gemm.md#wide-scalar-transposed-tiles

#include <src/linalg-impl.hh>
#include <src/sycl/device_scalar.hh>
#include <src/ops/gemm/gemm_kernels.hh>
#include <src/ops/gemm/kernels/copy_async.hh>
#include <src/ops/gemm/kernels/register_64x64_k16_wide.hh>
#include <src/sycl/kernel_attrs.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::sycl_gemm {

// One row of the wide-scalar transposed dispatch table, in the shape RegTile
// has for the float family. Structural, so it can be an NTTP.
struct WideTile {
    int M;                                  // macro tile rows
    int N;                                  // macro tile cols
    int K = 16;                             // k step
    int TTM = 4;                            // thread tile rows
    int TTN = 4;                            // thread tile cols
    Transpose OpA = Transpose::NoTrans;
    Transpose OpB = Transpose::NoTrans;
    int Stages = 1;                         // shared-memory stages: og::wide_stages
};

template <typename T, int M, int N, int K, int TTM, int TTN, int OA, int OB, int Stages, bool Vec, bool Split,
          int Async>
class GemmWideTransposedKernel;

// Split-k: group z of batch * splits computes item z / splits over k in [slice kc, slice kc + kc)
// and stores item z of C (the partials, alpha = 1, beta = 0; gemm_splitk reduces them).
struct WideSplit {
    int splits = 1;
    int kc = 0;
};

namespace wide_transposed {

using batchlas::sycl_device::Cx;
using batchlas::sycl_device::DevMap;
using batchlas::sycl_device::dev_conj;
using batchlas::sycl_device::dev_is_zero;
using batchlas::sycl_device::fma_acc;
using wide_scalar::lin_epi;
using wide_scalar::raw16_load;
using wide_scalar::raw16_store;
using wide_scalar::staged_c_store;
using wide_scalar::Vec16;
using wide_scalar::vec_ref;

// A fragment slice of W consecutive elements out of the shared tile. The
// 16-byte form is taken only when the slice IS the whole granule: at W < VecN
// the element offset `lane * W` need not be a multiple of VecN, so a Vec16 read
// there would be MISALIGNED, not merely wasteful. The 64x64 kernel's thread
// tile is 4 in both directions, so W == VecN always and it hard-wires it.
template <int W, int VecN, typename D>
inline void load_fragment(D* dst, const D* src) {
    if constexpr (W == VecN) {
        const Vec16<D> packet = vec_ref(src);
BATCHLAS_UNROLL_FULL
        for (int w = 0; w < W; ++w) {
            dst[w] = packet.v[w];
        }
    } else {
BATCHLAS_UNROLL_FULL
        for (int w = 0; w < W; ++w) {
            dst[w] = src[w];
        }
    }
}

// Staging moves 16-byte granules along the operand's contiguous dimension: k when KC (A^T, B),
// i (m for A, n for B) otherwise. Shared layout stays s[kk][i], TI wide. The KC store is a
// transposing scatter, so its granule column is XORed per row group: a conflict phase then
// covers 32 / TileK adjacent granules x distinct row groups on distinct banks. The XOR is a
// bijection within a row and the fragment read applies the same one.
// VecN == 1 (complex<double>) has no XOR: its fragment reads are one element per band, and the
// per-kk XORed addresses are loop-invariant, so they were hoisted out of the slab loop (one
// register per kk x band) and spilled. Its KC mapping below is conflict-free unswizzled instead.
// evidence: docs/perf/gemm.md#gemm-wp-k-pipelined-staging-for-the-transposed-wide-tiles
template <bool KC, int TI, int TileK, int VecN>
constexpr int soff(int kk, int i) {
    constexpr int GPR = TI / VecN;
    const int x = (KC && VecN > 1) ? ((kk / VecN) * (32 / TileK)) & (GPR - 1) : 0;
    return kk * TI + ((i / VecN) ^ x) * VecN + i % VecN;
}

// Granule g of a slab -> its first element (i, kk). KC: a warp takes 32 / KG consecutive i of
// KG granules each (whole TileK runs of 32 / KG columns, i fastest across lanes); otherwise
// consecutive granules of one shared row. KC at VecN == 1: a warp takes 16 i x 2 kk, lanes l and
// l + 8 one 32-byte sector, each 8-lane store phase 8 consecutive i of one shared row.
template <bool KC, int TI, int TileK, int VecN>
inline void granule(int g, int& i, int& kk) {
    if constexpr (KC && VecN == 1) {
        static_assert(TI % 16 == 0 && TileK % 2 == 0, "16 i x 2 kk per warp");
        const int w = g / 32;
        const int l = g % 32;
        i = (w % (TI / 16)) * 16 + l % 8 + 8 * (l / 16);
        kk = (w / (TI / 16)) * 2 + (l / 8) % 2;
    } else if constexpr (KC) {
        constexpr int IL = 32 / (TileK / VecN);
        i = (g / 32) * IL + (g % 32) % IL;
        kk = ((g % 32) / IL) * VecN;
    } else {
        i = (g % (TI / VecN)) * VecN;
        kk = g / (TI / VecN);
    }
}

// One granule into r[0..VecN): a 16-byte move on the Vec leg when the group's slab is full,
// else element loads, zero outside (i_lim, k) unless full.
template <bool KC, bool Vec, typename D>
inline void gload(D* r, const D* base, int ld, int gi, int gk, int i_lim, int k, bool full) {
    constexpr int VecN = Vec16<D>::N;
    const D* p = base + (KC ? static_cast<std::ptrdiff_t>(gi) * ld + gk : gi + static_cast<std::ptrdiff_t>(gk) * ld);
    if constexpr (Vec && VecN == 1) {
        // One guarded load, not a full/predicated diamond: ptxas if-converts it, so the slab's
        // FMAs (sunk below the staging) can be hoisted back. The diamond spilled every split tile.
        if (full || (gi < i_lim && gk < k)) raw16_store(r, raw16_load(p));
        else r[0] = D{};
        return;
    } else if constexpr (Vec) {
        if (full) {
            raw16_store(r, raw16_load(p));
            return;
        }
    }
BATCHLAS_UNROLL_FULL
    for (int q = 0; q < VecN; ++q) {
        const bool ok = full || (KC ? gi < i_lim && gk + q < k : gi + q < i_lim && gk < k);
        r[q] = ok ? p[q] : D{};
    }
}

template <bool KC, bool Conj, int TI, int TileK, typename D>
inline void sstore(D* s, const D* r, int i, int kk) {
    constexpr int VecN = Vec16<D>::N;
    alignas(16) D e[VecN];
BATCHLAS_UNROLL_FULL
    for (int q = 0; q < VecN; ++q) e[q] = Conj ? dev_conj(r[q]) : r[q];
    if constexpr (!KC) {
        raw16_store(s + soff<false, TI, TileK, VecN>(kk, i), raw16_load(e));
    } else {
BATCHLAS_UNROLL_FULL
        for (int q = 0; q < VecN; ++q) s[soff<true, TI, TileK, VecN>(kk + q, i)] = e[q];
    }
}

// One operand's slab straight into stage s (shared window address s_l), no registers and no conj
// (the fragment read applies it). Out-of-range elements are zero-filled with src-size 0 from the
// operand base. Along i (!KC) the 16-byte leg copies granules, src-size cut at the i edge; every
// other copy is one element, consecutive lanes on consecutive global addresses (along i, or along
// k when KC: the transposing scatter cannot be one 16-byte copy). Elements must sit at a multiple
// of their size (the launcher checks complex<float> bases; alignof is 4).
// evidence: docs/perf/gemm.md#gemm-wp-x-copy-async-stages-for-the-transposed-wide-tiles
template <bool KC, bool Vec, int TI, int TileK, int Threads, typename D>
// slab: the operand's element (i0, k0), inside it since i0 < i_lim and k0 < k_lim; i_rem and k_rem
// the extents left from there. Offsets are int: the launcher holds 128 ld to INT_MAX.
inline void acopy(D* s, std::uint32_t s_l, const D* slab, int ld, int i_rem, int k_rem, int tid) {
    constexpr int VecN = Vec16<D>::N;
    constexpr int ES = static_cast<int>(sizeof(D));
    if constexpr (Vec && !KC) {
        constexpr int Count = TI * TileK / VecN;
BATCHLAS_UNROLL_FULL
        for (int r = 0; r < (Count + Threads - 1) / Threads; ++r) {
            const int g = tid + r * Threads;
            if (Count % Threads != 0 && g >= Count) continue;
            const int i = (g % (TI / VecN)) * VecN, kk = g / (TI / VecN);
            const int left = i_rem - i;
            const bool ok = kk < k_rem && left > 0;
            const D* src = slab + (ok ? i + kk * ld : 0);
            const int o = soff<false, TI, TileK, VecN>(kk, i);
            copy_async_zfill<16>(s + o, s_l + static_cast<std::uint32_t>(ES * o), src,
                                 ok ? static_cast<std::uint32_t>(ES * (left < VecN ? left : VecN)) : 0u);
        }
    } else {
        constexpr int Count = TI * TileK;
        static_assert(Count % Threads == 0, "whole elements per lane");
BATCHLAS_UNROLL_FULL
        for (int r = 0; r < Count / Threads; ++r) {
            const int e = tid + r * Threads;
            const int i = KC ? e / TileK : e % TI, kk = KC ? e % TileK : e / TI;
            const bool ok = i < i_rem && kk < k_rem;
            const D* src = slab + (!ok ? 0 : KC ? i * ld + kk : i + kk * ld);
            const int o = soff<KC, TI, TileK, VecN>(kk, i);
            copy_async_zfill<ES>(s + o, s_l + static_cast<std::uint32_t>(ES * o), src,
                                 ok ? static_cast<std::uint32_t>(ES) : 0u);
        }
    }
}

}  // namespace wide_transposed

// Does an instantiation's hard-wired transpose form serve the requested one?
// A <ConjTrans, NoTrans> launcher handed a Trans operand silently drops the
// conjugation and returns a plausible wrong matrix, so call sites gate on the
// exact form -- with ONE licensed widening: for a real scalar conj is the
// identity, so ConjTrans and Trans instantiate each other. A real scalar
// compiles only the ConjTrans builds (choice.hh wide_serves); a complex one
// compiles Trans and ConjTrans apart and must match exactly.
template <typename T>
inline bool wide_trans_matches(Transpose requested, Transpose instantiated) {
    if (requested == instantiated) {
        return true;
    }
    if constexpr (is_std_complex_v<T>) {
        return false;
    } else {
        const bool both_transposing = requested != Transpose::NoTrans &&
            instantiated != Transpose::NoTrans;
        return both_transposing;
    }
}

// Two shared stages (P.Stages == 2) take one barrier per slab: slab s+1 is stored into the
// stage slab s-1 was read from, and every read of it ended before the barrier closing s-1.
// The next slab is fetched into registers before the current one's FMAs, except on the 64x64
// tiles (double and complex<double>), whose registers sit at the two-group cap already.
// evidence: docs/perf/gemm.md#gemm-wp-k-pipelined-staging-for-the-transposed-wide-tiles
// Async > 0: that many cp.async stages instead (one barrier per slab, Async - 1 slabs in flight).
template <typename T, WideTile P, bool VecLoads, bool Split, int Async = 0>
Event launch_wide_transposed_leg(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& B,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 T alpha,
                                 T beta,
                                 const char* trace,
                                 WideSplit split) {
    BATCHLAS_KERNEL_TRACE_SCOPE(trace);

    using namespace wide_transposed;

    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int TileM = P.M;
    constexpr int TileN = P.N;
    constexpr int TileK = P.K;
    constexpr int TTM = P.TTM;
    constexpr int TTN = P.TTN;
    constexpr int Stages = Async > 0 ? Async : P.Stages;
    // complex<double> prefetch spilled every tile (255 registers, ~4.4 KB, sm_120 probe).
    constexpr bool Prefetch = Async == 0 && !(TileM == 64 && TileN == 64) && sizeof(D) < 16;
    constexpr bool TransA = P.OpA != Transpose::NoTrans;
    constexpr bool TransB = P.OpB != Transpose::NoTrans;
    constexpr bool ConjA = P.OpA == Transpose::ConjTrans;
    constexpr bool ConjB = P.OpB == Transpose::ConjTrans;
    constexpr bool KCA = TransA;   // A^T is stored k x m: k contiguous
    constexpr bool KCB = !TransB;  // B is stored k x n: k contiguous

    constexpr int LocalRows = TileM / TTM;
    constexpr int LocalCols = TileN / TTN;
    constexpr int Threads = LocalRows * LocalCols;

    // No padding, for the reason the 64x64 kernel gives: an odd shared stride
    // defeats the compiler's alignment proof and fragment loads go scalar.
    constexpr int AStride = TileM;
    constexpr int BStride = TileN;
    constexpr int BufA = TileK * AStride;
    constexpr int BufB = TileK * BStride;

    constexpr int VecN = Vec16<D>::N;
    constexpr int WbM = (VecN < TTM) ? VecN : TTM;
    constexpr int WbN = (VecN < TTN) ? VecN : TTN;
    constexpr int NBM = TTM / WbM;
    constexpr int NBN = TTN / WbN;
    constexpr int MSep = TileM / NBM;
    constexpr int NSep = TileN / NBN;
    constexpr int CountA = TileM * TileK / VecN;  // granules per slab
    constexpr int CountB = TileN * TileK / VecN;
    constexpr int NGA = (CountA + Threads - 1) / Threads;
    constexpr int NGB = (CountB + Threads - 1) / Threads;

    static_assert(TileM % TTM == 0 && TileN % TTN == 0,
                  "macro tile must split into thread tiles");
    static_assert(TTM % WbM == 0 && TTN % WbN == 0,
                  "thread tile must split into whole bands");
    static_assert(Threads >= 32 && Threads <= 1024 && Threads % 32 == 0, "work-group size out of range");
    static_assert(Async > 0 ? Stages >= 2 && Stages <= 4 : Stages == 1 || Stages == 2, "shared stages");
    static_assert(TileK % VecN == 0 && TileK <= 32 && 32 % (TileK / VecN) == 0,
                  "a warp covers whole k runs of the k-contiguous operand");
    static_assert(CountA % 32 == 0 && CountB % 32 == 0, "a slab is whole warps of granules");
    // Single precision: the staged C epilogue in one pass and one round trip (each thread one row,
    // every prior load of its columns in flight), as wide v2's. Double precision keeps two per trip.
    // evidence: docs/perf/gemm.md#gemm-wp-x-copy-async-stages-for-the-transposed-wide-tiles
    constexpr bool OneRound = !std::is_same_v<D, double> && sizeof(D) <= 8 && Threads % TileM == 0 &&
        Stages * (BufA + BufB) >= TileM * TileN;
    constexpr int EpiLo = !OneRound ? Stages * BufA : Stages * BufA < TileM * TileN ? Stages * BufA : TileM * TileN;
    constexpr int EpiHi = OneRound ? TileM * TileN - EpiLo : Stages * BufB;
    constexpr int EpiBatch = OneRound ? TileN / (Threads / TileM) : 2;
    static_assert(EpiHi <= Stages * BufB && (!OneRound || EpiBatch > 2), "the one-round epilogue fits the stages");
    static_assert(WbM != VecN || (AStride % VecN == 0 && MSep % VecN == 0),
                  "vectorised A fragment needs an aligned band base");
    static_assert(WbN != VecN || (BStride % VecN == 0 && NSep % VecN == 0),
                  "vectorised B fragment needs an aligned band base");

    // Logical extents: A is stored k x m when OpA transposes it and B is stored
    // n x k when OpB does, but every bound below is in LOGICAL indices, so the
    // layouts differ only in the address formed.
    const int m = static_cast<int>(TransA ? A.cols() : A.rows());
    const int k = static_cast<int>(TransA ? A.rows() : A.cols());
    const int n = static_cast<int>(TransB ? B.rows() : B.cols());

    const int gm_tiles = (m + TileM - 1) / TileM;
    const int gn_tiles = (n + TileN - 1) / TileN;

    const sycl::range<3> local(1, LocalCols, LocalRows);
    const int splits = Split ? split.splits : 1;
    const int kc = split.kc;
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()) * static_cast<size_t>(splits),
                                static_cast<size_t>(gn_tiles) * LocalCols,
                                static_cast<size_t>(gm_tiles) * LocalRows);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<Vec16<D>, 1> tile_a(
            sycl::range<1>(static_cast<size_t>(Stages) * BufA / VecN), h);
        sycl::local_accessor<Vec16<D>, 1> tile_b(
            sycl::range<1>(static_cast<size_t>(Stages) * BufB / VecN), h);

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

        batchlas::impl::parallel_for<GemmWideTransposedKernel<T, TileM, TileN, TileK, TTM, TTN,
                                                static_cast<int>(P.OpA),
                                                static_cast<int>(P.OpB), Stages, VecLoads, Split, Async>>(h,
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int z = static_cast<int>(item.get_group(0));
                int bid = z;
                int k_lo = 0;
                int k_hi = k;
                if constexpr (Split) {
                    bid = z / splits;
                    k_lo = (z - bid * splits) * kc;
                    k_hi = sycl::min(k, k_lo + kc);
                }
                if (bid >= batch) {
                    return;
                }

                // Dimension 2 is fastest-varying and carries m: lanes one
                // apart touch consecutive addresses in column-major C.
                const int ty = static_cast<int>(item.get_local_id(2));
                const int tx = static_cast<int>(item.get_local_id(1));
                const int tid = tx * LocalRows + ty;

                const int m0 = static_cast<int>(item.get_group(2)) * TileM;
                const int n0 = static_cast<int>(item.get_group(1)) * TileN;
                const bool a_in = m0 + TileM <= m;
                const bool b_in = n0 + TileN <= n;

                const D* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                const D* Bb = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
                D* Cb = c_ptr + static_cast<std::ptrdiff_t>(z) * stride_c;

                D* sa = reinterpret_cast<D*>(
                    tile_a.template get_multi_ptr<sycl::access::decorated::no>().get());
                D* sb = reinterpret_cast<D*>(
                    tile_b.template get_multi_ptr<sycl::access::decorated::no>().get());

                D accum[TTM][TTN];
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < TTM; ++i) {
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < TTN; ++j) {
                        accum[i][j] = D{};
                    }
                }

                alignas(16) D ra[NGA * VecN];
                alignas(16) D rb[NGB * VecN];
                auto load = [&](int k0) __attribute__((always_inline)) {
                    const bool kin = k0 + TileK <= k_hi;
BATCHLAS_UNROLL_FULL
                    for (int r = 0; r < NGA; ++r) {
                        const int g = tid + r * Threads;
                        if (CountA % Threads != 0 && g >= CountA) continue;
                        int i, kk;
                        granule<KCA, TileM, TileK, VecN>(g, i, kk);
                        gload<KCA, VecLoads>(&ra[r * VecN], Ab, lda, m0 + i, k0 + kk, m, k_hi, a_in && kin);
                    }
BATCHLAS_UNROLL_FULL
                    for (int r = 0; r < NGB; ++r) {
                        const int g = tid + r * Threads;
                        if (CountB % Threads != 0 && g >= CountB) continue;
                        int i, kk;
                        granule<KCB, TileN, TileK, VecN>(g, i, kk);
                        gload<KCB, VecLoads>(&rb[r * VecN], Bb, ldb, n0 + i, k0 + kk, n, k_hi, b_in && kin);
                    }
                };
                auto store = [&](int stage) __attribute__((always_inline)) {
BATCHLAS_UNROLL_FULL
                    for (int r = 0; r < NGA; ++r) {
                        const int g = tid + r * Threads;
                        if (CountA % Threads != 0 && g >= CountA) continue;
                        int i, kk;
                        granule<KCA, TileM, TileK, VecN>(g, i, kk);
                        sstore<KCA, ConjA, TileM, TileK>(sa + stage * BufA, &ra[r * VecN], i, kk);
                    }
BATCHLAS_UNROLL_FULL
                    for (int r = 0; r < NGB; ++r) {
                        const int g = tid + r * Threads;
                        if (CountB % Threads != 0 && g >= CountB) continue;
                        int i, kk;
                        granule<KCB, TileN, TileK, VecN>(g, i, kk);
                        sstore<KCB, ConjB, TileN, TileK>(sb + stage * BufB, &rb[r * VecN], i, kk);
                    }
                };
                // Without prefetch each granule goes straight to its stage.
                auto stage_direct = [&](int k0, int stage) __attribute__((always_inline)) {
                    const bool kin = k0 + TileK <= k_hi;
BATCHLAS_UNROLL_FULL
                    for (int r = 0; r < NGA; ++r) {
                        const int g = tid + r * Threads;
                        if (CountA % Threads != 0 && g >= CountA) continue;
                        int i, kk;
                        granule<KCA, TileM, TileK, VecN>(g, i, kk);
                        alignas(16) D t[VecN];
                        gload<KCA, VecLoads>(t, Ab, lda, m0 + i, k0 + kk, m, k_hi, a_in && kin);
                        sstore<KCA, ConjA, TileM, TileK>(sa + stage * BufA, t, i, kk);
                    }
BATCHLAS_UNROLL_FULL
                    for (int r = 0; r < NGB; ++r) {
                        const int g = tid + r * Threads;
                        if (CountB % Threads != 0 && g >= CountB) continue;
                        int i, kk;
                        granule<KCB, TileN, TileK, VecN>(g, i, kk);
                        alignas(16) D t[VecN];
                        gload<KCB, VecLoads>(t, Bb, ldb, n0 + i, k0 + kk, n, k_hi, b_in && kin);
                        sstore<KCB, ConjB, TileN, TileK>(sb + stage * BufB, t, i, kk);
                    }
                };
                // Out-of-range staging wrote a zero rather than being skipped,
                // so the accumulation needs no bounds test.
                auto compute = [&](int stage) __attribute__((always_inline)) {
                    const D* xa = sa + stage * BufA;
                    const D* xb = sb + stage * BufB;
BATCHLAS_UNROLL_FULL
                    for (int kk = 0; kk < TileK; ++kk) {
                        D af[TTM];
                        D bf[TTN];
BATCHLAS_UNROLL_FULL
                        for (int band = 0; band < NBM; ++band) {
                            load_fragment<WbM, VecN>(
                                &af[band * WbM], xa + soff<KCA, TileM, TileK, VecN>(kk, band * MSep + ty * WbM));
                        }
BATCHLAS_UNROLL_FULL
                        for (int band = 0; band < NBN; ++band) {
                            load_fragment<WbN, VecN>(
                                &bf[band * WbN], xb + soff<KCB, TileN, TileK, VecN>(kk, band * NSep + tx * WbN));
                        }
                        if constexpr (Async > 0 && ConjA) {
BATCHLAS_UNROLL_FULL
                            for (int i = 0; i < TTM; ++i) af[i] = dev_conj(af[i]);
                        }
                        if constexpr (Async > 0 && ConjB) {
BATCHLAS_UNROLL_FULL
                            for (int j = 0; j < TTN; ++j) bf[j] = dev_conj(bf[j]);
                        }
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < TTM; ++i) {
BATCHLAS_UNROLL_FULL
                            for (int j = 0; j < TTN; ++j) {
                                fma_acc(accum[i][j], af[i], bf[j]);
                            }
                        }
                    }
                };

                const int slabs = (k_hi - k_lo + TileK - 1) / TileK;
                if constexpr (Async > 0) {
                    // As wide v2's 128x128 pipeline: step s waits for slab s (own copies, then the
                    // barrier for everyone's), refills the stage slab s - 1 was read from, computes.
                    // One commit per step, empty past the end, keeps the wait count at Stages - 2.
                    std::uint32_t sa_l = 0, sb_l = 0;
#if BATCHLAS_GEMM_CP_ASYNC
                    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(sa_l) : "l"(sa));
                    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(sb_l) : "l"(sb));
#endif
                    constexpr std::uint32_t ES = sizeof(D);
                    auto issue = [&](int k0, int buf) __attribute__((always_inline)) {
                        const D* pa = Ab + (KCA ? static_cast<std::ptrdiff_t>(m0) * lda + k0
                                                : m0 + static_cast<std::ptrdiff_t>(k0) * lda);
                        const D* pb = Bb + (KCB ? static_cast<std::ptrdiff_t>(n0) * ldb + k0
                                                : n0 + static_cast<std::ptrdiff_t>(k0) * ldb);
                        acopy<KCA, VecLoads, TileM, TileK, Threads>(sa + buf * BufA, sa_l + ES * (buf * BufA), pa, lda,
                                                                    m - m0, k_hi - k0, tid);
                        acopy<KCB, VecLoads, TileN, TileK, Threads>(sb + buf * BufB, sb_l + ES * (buf * BufB), pb, ldb,
                                                                    n - n0, k_hi - k0, tid);
                    };
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < Stages - 1; ++j) {
                        if (j < slabs) issue(k_lo + j * TileK, j);
                        copy_async_commit();
                    }
                    int buf = 0;
#pragma unroll 1
                    for (int s = 0; s < slabs; ++s) {
                        copy_async_wait<Stages - 2>();
                        item.barrier(sycl::access::fence_space::local_space);
                        const int nb = buf == 0 ? Stages - 1 : buf - 1;
                        if (s + Stages - 1 < slabs) issue(k_lo + (s + Stages - 1) * TileK, nb);
                        copy_async_commit();
                        compute(buf);
                        buf = buf + 1 == Stages ? 0 : buf + 1;
                    }
                    copy_async_wait<0>();
                } else {
                if constexpr (Prefetch) {
                    load(k_lo);
                    store(0);
                } else {
                    stage_direct(k_lo, 0);
                }
                item.barrier(sycl::access::fence_space::local_space);
                for (int s = 0; s < slabs; ++s) {
                    const int cur = Stages == 2 ? (s & 1) : 0;
                    const bool more = s + 1 < slabs;
                    if constexpr (Prefetch) {
                        if (more) load(k_lo + (s + 1) * TileK);
                    }
                    compute(cur);
                    if (more) {
                        if constexpr (Stages == 1) item.barrier(sycl::access::fence_space::local_space);
                        if constexpr (Prefetch) store(Stages == 2 ? cur ^ 1 : 0);
                        else stage_direct(k_lo + (s + 1) * TileK, Stages == 2 ? cur ^ 1 : 0);
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                }
                }

                // Per group: the 16-byte band store when this tile is inside
                // C with 16-byte columns and a band is one granule; otherwise
                // staged through sa/sb (lanes WbM rows apart would store one
                // element each), except at VecN == 1, where that scalar store
                // is the band store. Same rule as the 64x64 NN kernel.
                const bool beta_zero = dev_is_zero(beta_d);
                if constexpr (VecN > 1) {
                    if constexpr (WbM == VecN) {
                        const bool c_vec = m0 + TileM <= m && n0 + TileN <= n &&
                            ldc % VecN == 0 && reinterpret_cast<std::uintptr_t>(Cb) % 16u == 0;
                        if (c_vec) {
BATCHLAS_UNROLL_FULL
                            for (int bm = 0; bm < NBM; ++bm) {
                                const int row = m0 + bm * MSep + ty * WbM;
BATCHLAS_UNROLL_FULL
                                for (int j = 0; j < TTN; ++j) {
                                    const int col = n0 + (j / WbN) * NSep + tx * WbN + j % WbN;
                                    D* p = &Cb[row + static_cast<std::ptrdiff_t>(col) * ldc];
                                    Vec16<D> out;
                                    Vec16<D> prior{};
                                    if (!beta_zero) prior = vec_ref(const_cast<const D*>(p));
BATCHLAS_UNROLL_FULL
                                    for (int w = 0; w < VecN; ++w) {
                                        out.v[w] = lin_epi(alpha_d, beta_zero ? D{} : beta_d,
                                                           accum[bm * WbM + w][j], prior.v[w]);
                                    }
                                    vec_ref(p) = out;
                                }
                            }
                            return;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);  // the last compute's reads
                    staged_c_store<TileM, TileN, Threads, EpiLo, EpiHi, EpiBatch>(
                        item, sa, sb, Cb, ldc, m0, n0, m, n, tid, alpha_d, beta_d, beta_zero,
                        [&](int c0, int cw, auto slot) {
BATCHLAS_UNROLL_FULL
                            for (int bm = 0; bm < NBM; ++bm) {
BATCHLAS_UNROLL_FULL
                                for (int j = 0; j < TTN; ++j) {
                                    const int c = (j / WbN) * NSep + tx * WbN + j % WbN - c0;
                                    if (c < 0 || c >= cw) continue;
                                    D* s = slot(c, bm * MSep + ty * WbM);
                                    if constexpr (WbM == VecN) {
                                        Vec16<D> v;
BATCHLAS_UNROLL_FULL
                                        for (int w = 0; w < WbM; ++w) v.v[w] = accum[bm * WbM + w][j];
                                        vec_ref(s) = v;
                                    } else {
BATCHLAS_UNROLL_FULL
                                        for (int w = 0; w < WbM; ++w) s[w] = accum[bm * WbM + w][j];
                                    }
                                }
                            }
                        });
                    return;
                }
BATCHLAS_UNROLL_FULL
                for (int bm = 0; bm < NBM; ++bm) {
BATCHLAS_UNROLL_FULL
                    for (int w = 0; w < WbM; ++w) {
                        const int row = m0 + bm * MSep + ty * WbM + w;
                        if (row >= m) {
                            continue;
                        }
BATCHLAS_UNROLL_FULL
                        for (int bn = 0; bn < NBN; ++bn) {
BATCHLAS_UNROLL_FULL
                            for (int v = 0; v < WbN; ++v) {
                                const int col = n0 + bn * NSep + tx * WbN + v;
                                if (col >= n) {
                                    continue;
                                }
                                D* p = &Cb[row + static_cast<std::ptrdiff_t>(col) * ldc];
                                const D prior = beta_zero ? D{} : *p;
                                *p = lin_epi(alpha_d, beta_zero ? D{} : beta_d,
                                             accum[bm * WbM + w][bn * WbN + v], prior);
                            }
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

// The leg (16-byte or element global loads) is wide v2's predicate, per launch; whether a slab
// is predicated is per work-group. complex<float> adds the cp.async leg (trace_async) where DPC++
// emits cp.async, at sm_120+ (the only architecture timed; BATCHLAS_GEMM_WIDE_STAGES=3 admits sm_80+,
// =2 forces the register prefetch) and both bases are 8-byte aligned (alignof is 4). Two stages:
// three and four lost occupancy; float's 4-byte copies cost 16-45 registers and ran slower.
// evidence: docs/perf/gemm.md#gemm-wp-x-copy-async-stages-for-the-transposed-wide-tiles
template <typename T, WideTile P, bool Split = false>
Event launch_wide_transposed(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C,
                             T alpha, T beta, const char* trace, WideSplit split = {},
                             const char* trace_async = nullptr) {
    if constexpr (std::is_same_v<T, std::complex<float>>) {
        constexpr int Async = 2;
        constexpr int Threads = (P.M / P.TTM) * (P.N / P.TTN);
        constexpr std::size_t bytes = std::size_t(Async) * P.K * (P.M + P.N) * sizeof(T);
        const int forced = batchlas::settings().geometry.gemm_wide_stages;
        auto whole = [](const T* p) { return reinterpret_cast<std::uintptr_t>(p) % sizeof(T) == 0; };
        if (!BATCHLAS_SYCL_IMPL_ACPP && forced != 2 &&
            ctx.device().cuda_compute_capability() >= (forced == 3 ? 80 : 120) && whole(A.data_ptr()) &&
            whole(B.data_ptr()) && std::max(A.ld(), B.ld()) <= INT_MAX / 128 &&
            resident::device_slm_budget(impl::local_mem_bytes(ctx->get_device(), Threads)) >= bytes) {
            const char* t = trace_async ? trace_async : trace;
            if (wide64_v2_vec_loads<T>(A, B))
                return launch_wide_transposed_leg<T, P, true, Split, Async>(ctx, A, B, C, alpha, beta, t, split);
            return launch_wide_transposed_leg<T, P, false, Split, Async>(ctx, A, B, C, alpha, beta, t, split);
        }
    }
    if (wide64_v2_vec_loads<T>(A, B))
        return launch_wide_transposed_leg<T, P, true, Split>(ctx, A, B, C, alpha, beta, trace, split);
    return launch_wide_transposed_leg<T, P, false, Split>(ctx, A, B, C, alpha, beta, trace, split);
}

}  // namespace batchlas::sycl_gemm

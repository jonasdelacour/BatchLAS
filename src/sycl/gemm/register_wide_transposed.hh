#pragma once

// A register-tiled GEMM for TRANSPOSED and CONJUGATE-TRANSPOSED operands with
// a wide scalar (double, complex<float>, complex<double>).
//
// It exists because neither existing family can serve these shapes correctly
// AND fast: register_tiled_common.hh carries Transpose OpA/OpB but multiplies
// through std::complex operator*, which is Annex-G conformant and costs an
// isnan branch plus a __mulsc3 / __muldc3 call per multiply (see
// ../device_scalar.hh); register_64x64_k16_wide.hh keeps std::complex out of
// device code but reads A as m x k and B as k x n and is NN only.
//
// The macro tile is a PARAMETER, not 64x64, because the demand is not square:
// every complex transposed shape the blocked drivers issue has a dimension of
// 16 or 32 (a panel width), so a square tile wastes half or three quarters of
// itself. The tile-to-driver mapping, the shapes, what is deliberately not
// built and the grid that must be run before any selector row opens are all in
// evidence: docs/perf/gemm.md#wide-scalar-transposed-tiles

#include "../../linalg-impl.hh"
#include "../device_scalar.hh"
#include "../gemm_kernels.hh"
#include "register_64x64_k16_wide.hh"

#include <sycl/sycl.hpp>

#include <cstddef>
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
};

template <typename T, int M, int N, int K, int TTM, int TTN, int OA, int OB>
class GemmWideTransposedKernel;

namespace wide_transposed {

using batchlas::sycl_device::Cx;
using batchlas::sycl_device::DevMap;
using batchlas::sycl_device::dev_conj;
using batchlas::sycl_device::dev_is_zero;
using batchlas::sycl_device::fma_acc;
using wide_scalar::lin_epi;
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
#pragma unroll
        for (int w = 0; w < W; ++w) {
            dst[w] = packet.v[w];
        }
    } else {
#pragma unroll
        for (int w = 0; w < W; ++w) {
            dst[w] = src[w];
        }
    }
}

}  // namespace wide_transposed

// Does an instantiation's hard-wired transpose form serve the requested one?
// A <ConjTrans, NoTrans> launcher handed a Trans operand silently drops the
// conjugation and returns a plausible wrong matrix, so call sites gate on the
// exact form -- with ONE licensed widening: for a real scalar conj is the
// identity, so ConjTrans and Trans instantiate each other. That widening is
// what lets one variant serve potrf_blocked.cc's kTrailingTransB<T>.
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

template <typename T, WideTile P>
Event launch_wide_transposed(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             T alpha,
                             T beta,
                             const char* trace) {
    BATCHLAS_KERNEL_TRACE_SCOPE(trace);

    using namespace wide_transposed;

    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int TileM = P.M;
    constexpr int TileN = P.N;
    constexpr int TileK = P.K;
    constexpr int TTM = P.TTM;
    constexpr int TTN = P.TTN;
    constexpr bool TransA = P.OpA != Transpose::NoTrans;
    constexpr bool TransB = P.OpB != Transpose::NoTrans;
    constexpr bool ConjA = P.OpA == Transpose::ConjTrans;
    constexpr bool ConjB = P.OpB == Transpose::ConjTrans;

    constexpr int LocalRows = TileM / TTM;
    constexpr int LocalCols = TileN / TTN;
    constexpr int Threads = LocalRows * LocalCols;

    // No padding, for the reason the 64x64 kernel gives: an odd shared stride
    // defeats the compiler's alignment proof and fragment loads go scalar.
    constexpr int AStride = TileM;
    constexpr int BStride = TileN;

    constexpr int VecN = Vec16<D>::N;
    constexpr int WbM = (VecN < TTM) ? VecN : TTM;
    constexpr int WbN = (VecN < TTN) ? VecN : TTN;
    constexpr int NBM = TTM / WbM;
    constexpr int NBN = TTN / WbN;
    constexpr int MSep = TileM / NBM;
    constexpr int NSep = TileN / NBN;

    static_assert(TileM % TTM == 0 && TileN % TTN == 0,
                  "macro tile must split into thread tiles");
    static_assert(TTM % WbM == 0 && TTN % WbN == 0,
                  "thread tile must split into whole bands");
    static_assert(Threads >= 32 && Threads <= 1024, "work-group size out of range");
    static_assert((TileK * AStride) % VecN == 0 && (TileK * BStride) % VecN == 0,
                  "shared tile must be a whole number of 16-byte granules");
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
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(gn_tiles) * LocalCols,
                                static_cast<size_t>(gm_tiles) * LocalRows);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<Vec16<D>, 1> tile_a(
            sycl::range<1>(static_cast<size_t>(TileK) * AStride / VecN), h);
        sycl::local_accessor<Vec16<D>, 1> tile_b(
            sycl::range<1>(static_cast<size_t>(TileK) * BStride / VecN), h);

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

        h.parallel_for<GemmWideTransposedKernel<T, TileM, TileN, TileK, TTM, TTN,
                                                static_cast<int>(P.OpA),
                                                static_cast<int>(P.OpB)>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
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

                const D* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                const D* Bb = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
                D* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;

                D* sa = reinterpret_cast<D*>(
                    tile_a.template get_multi_ptr<sycl::access::decorated::no>().get());
                D* sb = reinterpret_cast<D*>(
                    tile_b.template get_multi_ptr<sycl::access::decorated::no>().get());

                D accum[TTM][TTN];
#pragma unroll
                for (int i = 0; i < TTM; ++i) {
#pragma unroll
                    for (int j = 0; j < TTN; ++j) {
                        accum[i][j] = D{};
                    }
                }

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    // Staging always produces the SAME shared layout,
                    // sa[kk][i] and sb[kk][j], whatever the operand layout was,
                    // which keeps the inner loop and epilogue identical across
                    // all four forms. The index decomposition is per layout so
                    // consecutive `tid` reads consecutive global addresses: a
                    // transposed operand pays on the shared STORE, never on the
                    // global load.
                    for (int idx = tid; idx < TileM * TileK; idx += Threads) {
                        int i;
                        int kk;
                        if constexpr (TransA) {
                            kk = idx % TileK;
                            i = idx / TileK;
                        } else {
                            i = idx % TileM;
                            kk = idx / TileM;
                        }
                        const int gm = m0 + i;
                        const int gk = k0 + kk;
                        D value = D{};
                        if (gm < m && gk < k) {
                            value = TransA
                                ? Ab[gk + static_cast<std::ptrdiff_t>(gm) * lda]
                                : Ab[gm + static_cast<std::ptrdiff_t>(gk) * lda];
                            if constexpr (ConjA) {
                                value = dev_conj(value);
                            }
                        }
                        sa[kk * AStride + i] = value;
                    }

                    for (int idx = tid; idx < TileK * TileN; idx += Threads) {
                        int j;
                        int kk;
                        if constexpr (TransB) {
                            j = idx % TileN;
                            kk = idx / TileN;
                        } else {
                            kk = idx % TileK;
                            j = idx / TileK;
                        }
                        const int gn = n0 + j;
                        const int gk = k0 + kk;
                        D value = D{};
                        if (gn < n && gk < k) {
                            value = TransB
                                ? Bb[gn + static_cast<std::ptrdiff_t>(gk) * ldb]
                                : Bb[gk + static_cast<std::ptrdiff_t>(gn) * ldb];
                            if constexpr (ConjB) {
                                value = dev_conj(value);
                            }
                        }
                        sb[kk * BStride + j] = value;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Out-of-range staging wrote a zero rather than being
                    // skipped, so the accumulation needs no bounds test.
#pragma unroll
                    for (int kk = 0; kk < TileK; ++kk) {
                        D af[TTM];
                        D bf[TTN];
#pragma unroll
                        for (int band = 0; band < NBM; ++band) {
                            load_fragment<WbM, VecN>(
                                &af[band * WbM], &sa[kk * AStride + band * MSep + ty * WbM]);
                        }
#pragma unroll
                        for (int band = 0; band < NBN; ++band) {
                            load_fragment<WbN, VecN>(
                                &bf[band * WbN], &sb[kk * BStride + band * NSep + tx * WbN]);
                        }
#pragma unroll
                        for (int i = 0; i < TTM; ++i) {
#pragma unroll
                            for (int j = 0; j < TTN; ++j) {
                                fma_acc(accum[i][j], af[i], bf[j]);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Predicated scalar epilogue only: these tiles MATCH a
                // panel width, so the n edge is partial by construction at the
                // shapes the drivers issue and the 64x64 NN kernel's banded
                // 16-byte store would never fire. O(TileM * TileN) per group
                // against O(TileM * TileN * k) multiplies.
                const bool beta_zero = dev_is_zero(beta_d);
#pragma unroll
                for (int bm = 0; bm < NBM; ++bm) {
#pragma unroll
                    for (int w = 0; w < WbM; ++w) {
                        const int row = m0 + bm * MSep + ty * WbM + w;
                        if (row >= m) {
                            continue;
                        }
#pragma unroll
                        for (int bn = 0; bn < NBN; ++bn) {
#pragma unroll
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

}  // namespace batchlas::sycl_gemm

#pragma once

#include "../util/internal-api.hh"
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_gemm {

enum class KernelVariant {
    Direct,
    Tiled16,
    Tiled32x32Register,
    Tiled64x64Register,
    Tiled64x64RegisterK16,
    Tiled64x64RegisterK16TN,
    Tiled64x64RegisterK16NT,
    Tiled64x64RegisterK16TT,
    Tiled128x32RegisterK16,
    Tiled128x32RegisterK16TN,
    Tiled128x32RegisterK16NT,
    Tiled128x32RegisterK16TT,
    Tiled128x32RegisterK32TN,
    Tiled128x32RegisterK32NT,
    Tiled128x32RegisterK32TT,
    Tiled128x64RegisterK16TN,
    Tiled128x64RegisterK16NT,
    Tiled128x64RegisterK16TT,
    // Legacy alias kept for compatibility. The canonical 128x32x32 family
    // names below carry the actual compile-time stage/unroll parameters.
    Tiled128x32RegisterK32,
    Tiled128x32RegisterK32S1U1,
    Tiled128x32RegisterK32S2U1,
    Tiled128x32RegisterK32S2U1Aligned,
    Tiled128x32RegisterK32S2U1Generic,
    Tiled128x32RegisterK32S2U2,
    Tiled128x32RegisterK32S2U2TT8x4,
    Tiled128x32RegisterK32S2U2TT4x8,
    Tiled128x32RegisterK32Persistent,
    Tiled128x32RegisterK32SplitK4,
    Tiled128x32RegisterK32S1U4,
    Tiled128x64RegisterK32Large,
    Tiled128x64RegisterK32LargeU2,
    Tiled128x64RegisterK32LargeTT4x8,
    Tiled128x64RegisterK32LargeTT4x8U2,
    // Shallow K, wide M and N: the only variant with a 64-accumulator thread
    // tile, and the only one whose shared fragment loads vectorize.
    Tiled128x128RegisterK8,
    // The wide-scalar tile: 4x4 thread tile, 16 accumulators, and a 16-byte
    // (not 4-element) access granule, so double / complex<float> /
    // complex<double> all get vectorized conflict-free fragment loads. The
    // only register-tiled variant that serves a non-float scalar.
    Tiled64x64RegisterK16Wide,
    // The transposed / conj-transposed wide-scalar family: macro tile matched to
    // a PANEL WIDTH. evidence: docs/perf/gemm.md#wide-scalar-transposed-tiles
    Tiled64x64RegisterK16WideCN,
    Tiled64x64RegisterK16WideNC,
    Tiled128x32RegisterK16WideNC,
    Tiled32x128RegisterK16WideCN,
    Tiled32x128RegisterK16,
    Tiled32x128RegisterK16TN,
    Tiled32x128RegisterK16TT,
};

// Which wide-scalar TRANSPOSED tile a shape fits: the matching macro-tile
// dimension must be FILLED, since these tiles are cut to a panel width rather
// than square. A selector predicate, never a routing one -- its arm is Tiled16.
// evidence: docs/perf/gemm.md#wide-scalar-transposed-tiles
enum class WideTransposedTile { None, NC128x32, CN32x128 };

inline WideTransposedTile wide_transposed_tile_for(Transpose transA, Transpose transB,
                                                   int64_t m, int64_t n, int64_t k,
                                                   int64_t batch) {
    if (k < 8) return WideTransposedTile::None;
    constexpr int64_t kMinCtas = 64;
    auto ctas = [&](int64_t tm, int64_t tn) {
        return ((m + tm - 1) / tm) * ((n + tn - 1) / tn) * batch;
    };
    // ConjTrans EXACTLY: these are ConjTrans instantiations and
    // wide_trans_matches refuses to serve a COMPLEX Trans with one, so a Trans
    // shape selected here would silently fall back and the selector would be
    // naming a kernel that did not run.
    if (transA == Transpose::NoTrans && transB == Transpose::ConjTrans) {
        if (m >= 128 && n >= 32 && ctas(128, 32) >= kMinCtas) {
            return WideTransposedTile::NC128x32;
        }
    } else if (transA == Transpose::ConjTrans && transB == Transpose::NoTrans) {
        if (m >= 32 && n >= 128 && ctas(32, 128) >= kMinCtas) {
            return WideTransposedTile::CN32x128;
        }
    }
    return WideTransposedTile::None;
}


template <typename T>
BATCHLAS_INTERNAL_API KernelVariant select_kernel_variant(const MatrixView<T, MatrixFormat::Dense>& A,
                                                          const MatrixView<T, MatrixFormat::Dense>& B,
                                                          const MatrixView<T, MatrixFormat::Dense>& C,
                                                          Transpose transA,
                                                          Transpose transB);

template <typename T>
Event gemm_custom(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  T alpha,
                  T beta,
                  Transpose transA,
                  Transpose transB,
                  ComputePrecision precision);

} // namespace batchlas::sycl_gemm
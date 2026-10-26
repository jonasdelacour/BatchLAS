#pragma once

// potri: the local-memory-resident trtri/lauum kernel, the info scan and the triangle finish.
// evidence: docs/perf/potri.md#potri-the-algorithm

#include <src/util/internal-api.hh>
#include <src/ops/potrf/potrf_slm_hole.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_potri {

// Potri: the uplo triangle of A := that of A^-1 from the factor (trtri then lauum, one launch).
// Trtri: every nb x nb diagonal block of A (the last one ragged) := its triangular inverse.
// Lauum: A := X^H X (Lower) or X X^H (Upper) from the triangular X in A's uplo triangle.
enum class CtaMode : int { Potri = 0, Trtri = 1, Lauum = 2 };

// Register tiles: TS x TS per tile, at most kCtaTiles tiles per work-item, at most kCtaMaxWg
// work-items; one tile per lower-triangle tile of the order-m block padded to a multiple of TS.
template <typename T>
inline constexpr int kCtaTs = sizeof(T) > 8 ? 2 : 4;
inline constexpr int kCtaTiles = 2;
inline constexpr int kCtaMaxWg = 512;

constexpr int cta_tiles(int m, int ts) {
    const int t = (m + ts - 1) / ts;
    return t * (t + 1) / 2;
}

// Work-items per block: one per tile up to the cap, a multiple of 32.
constexpr int cta_wg(int m, int ts, int max_wg) {
    const int cap = max_wg < kCtaMaxWg ? max_wg : kCtaMaxWg;
    const int want = (cta_tiles(m, ts) + 31) / 32 * 32;
    return want < cap ? want : cap;
}

// The padded tile (odd ld) and the padded reals. Shared by can_run (through cta_max_n) and the
// launcher, so the advertised ceiling is the allocated one.
constexpr std::size_t cta_slm_bytes(int m, int ts, std::size_t sz_d, std::size_t sz_r) {
    const std::size_t mp = static_cast<std::size_t>((m + ts - 1) / ts * ts);
    return potrf_native::potrf_hole_padded((mp + 1) * mp * sz_d + mp * sz_r + 64);
}

constexpr bool cta_fits(int m, int ts, std::size_t sz_d, std::size_t sz_r, std::size_t slm_budget, int max_wg) {
    return m >= 1 && cta_slm_bytes(m, ts, sz_d, sz_r) <= slm_budget &&
           cta_tiles(m, ts) <= kCtaTiles * cta_wg(m, ts, max_wg);
}

// The largest order that fits (and every smaller one does); 0 if none.
template <typename T>
constexpr int cta_max_n(std::size_t slm_budget, int max_wg) {
    using R = typename base_type<T>::type;
    int best = 0;
    for (int m = 1; m <= 1024 && cta_fits(m, kCtaTs<T>, sizeof(T), sizeof(R), slm_budget, max_wg); ++m) best = m;
    return best;
}

// mode Trtri inverts the ceil(n / nb) diagonal blocks of every item; Potri and Lauum ignore nb.
// info (Potri only, may be empty): 1-based first exactly-zero factor diagonal, else 0; such an
// item is left unchanged. full (Potri only) also writes the other triangle (conjugate transpose).
template <typename T>
BATCHLAS_INTERNAL_API Event potri_cta(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                      CtaMode mode, int nb, bool full, Span<int32_t> info);

// info[b] := 1-based first exactly-zero diagonal index, else 0.
template <typename T>
BATCHLAS_INTERNAL_API Event potri_diag_info(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                            Span<int32_t> info);

// The diagonal's imaginary part := 0; with full the other triangle := the uplo one's conjugate transpose.
template <typename T>
BATCHLAS_INTERNAL_API Event potri_finish(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                         bool full);

}  // namespace batchlas::sycl_potri

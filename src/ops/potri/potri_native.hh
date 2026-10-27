#pragma once

// potri's kernels: the CTA trtri/lauum and the finish. evidence: docs/perf/potri.md#potri-the-algorithm

#include <src/util/internal-api.hh>
#include <src/ops/potrf/potrf_slm_hole.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_potri {

// Potri: A's uplo triangle := A^-1's from the factor. Trtri: each nb x nb diagonal block := its inverse.
// Lauum: A := X^H X (Lower) or X X^H (Upper) from the triangular X in A's uplo triangle.
enum class CtaMode : int { Potri = 0, Trtri = 1, Lauum = 2 };

// TS x TS register tiles, one per lower-triangle tile, at most kCtaTiles per work-item of kCtaMaxWg.
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

// The padded tile (odd ld) and reals: can_run (cta_max_n) and the launcher share it.
constexpr std::size_t cta_slm_bytes(int m, int ts, std::size_t sz_d, std::size_t sz_r) {
    const std::size_t mp = static_cast<std::size_t>((m + ts - 1) / ts * ts);
    return potrf_native::potrf_hole_padded((mp + 1) * mp * sz_d + mp * sz_r + 64);
}

constexpr bool cta_fits(int m, int ts, std::size_t sz_d, std::size_t sz_r, std::size_t slm_budget, int max_wg) {
    return m >= 1 && cta_slm_bytes(m, ts, sz_d, sz_r) <= slm_budget &&
           cta_tiles(m, ts) <= kCtaTiles * cta_wg(m, ts, max_wg);
}

template <typename T>
constexpr int cta_max_n(std::size_t slm_budget, int max_wg) {
    using R = typename base_type<T>::type;
    int best = 0;
    for (int m = 1; m <= 1024 && cta_fits(m, kCtaTs<T>, sizeof(T), sizeof(R), slm_budget, max_wg); ++m) best = m;
    return best;
}

// blocked keeps a zero-diagonal item unchanged: Trtri sets flags[b] (1-based) and copies a flagged item
// to slot b (its packed uplo triangle); Lauum on the block at (origin, origin) copies it back.
template <typename T>
struct Stash {
    T* slots = nullptr;
    int32_t* flags = nullptr;
    int n = 0;
    int origin = 0;
};

// nb: Trtri's block order. info and full: Potri only; an item with an exactly zero diagonal entry
// gets its 1-based index and is left unchanged; full writes the conjugate into the other triangle.
template <typename T>
BATCHLAS_INTERNAL_API Event potri_cta(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                      CtaMode mode, int nb, bool full, Span<int32_t> info, Stash<T> stash = {});

// Diagonal imaginary part := 0; full: other triangle := uplo's conjugate transpose. Skips skip[b] != 0.
template <typename T>
BATCHLAS_INTERNAL_API Event potri_finish(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                         bool full, const int32_t* skip = nullptr);

}  // namespace batchlas::sycl_potri

// The blocked family (kernels/potri_blocked.cc): trtri then lauum on nb x nb blocks.
namespace batchlas::ops::potri::blocked {

template <class T>
std::size_t workspace(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, int nb);

template <Backend B, class T>
Event run(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, int nb, bool full,
          Span<std::byte> ws_span, Span<int32_t> info);

}  // namespace batchlas::ops::potri::blocked

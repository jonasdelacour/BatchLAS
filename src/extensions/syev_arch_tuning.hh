#pragma once

// Per-architecture defaults for syev's blocked and two-stage paths, keyed on
// Device::cuda_compute_capability(). Anything but sm_120 gets the sm_89 value;
// the BATCHLAS_TUNE_* / env overrides still win. Kept out of tuning_params.hh,
// which the tuning harness regenerates. evidence: docs/perf/blackwell.md#syev-retune

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/settings.hh>
#include <batchlas/tuning_params.hh>
#include <cstdint>

namespace batchlas::syev_tuning {

inline constexpr int32_t kSm120SytrdBlockXlarge = 16;
inline constexpr int32_t kSm120OrmqrBlockXlarge = 32;
inline constexpr int32_t kSm120LatrdLegacyMaxN[2][2] = {{320, 256}, {256, 128}};  // [complex][double]: largest legacy panel n
inline constexpr int32_t kSm89LatrdGridMinN = 768;

// syev_blocked's sytrd panel width, before its complex 256 < n <= 512 bucket.
inline constexpr int32_t sytrd_block_size_default_for_n(int32_t n, int cuda_cc) {
    if (dispatch::is_sm120_family(cuda_cc) && n > 512) return kSm120SytrdBlockXlarge;
    return tuning::sytrd_block_size_default_for_n(n);
}

inline int32_t sytrd_block_size_for_n(int32_t n, int cuda_cc) {
    return tuning::detail::tuning_env_override(settings().geometry.tune.sytrd_block_size,
                                               sytrd_block_size_default_for_n(n, cuda_cc));
}

// syev's own ORMQR_BLOCK_SIZE_*: ormqr, gesvd_blocked and syevx keep the shared one.
inline constexpr int32_t ormqr_block_size_default_for_n(int32_t n, int cuda_cc) {
    if (dispatch::is_sm120_family(cuda_cc) && n > 512) return kSm120OrmqrBlockXlarge;
    return tuning::ormqr_block_size_default_for_n(n);
}

inline int32_t ormqr_block_size_for_n(int32_t n, int cuda_cc) {
    return tuning::detail::tuning_env_override(settings().geometry.tune.ormqr_block_size,
                                               ormqr_block_size_default_for_n(n, cuda_cc));
}

// The n from which latrd's grid path is the default.
inline constexpr int32_t latrd_grid_min_n_default(int cuda_cc, bool is_complex, bool is_double) {
    if (!dispatch::is_sm120_family(cuda_cc)) return kSm89LatrdGridMinN;
    return kSm120LatrdLegacyMaxN[is_complex][is_double] + 1;
}

inline int32_t latrd_grid_min_n(int cuda_cc, bool is_complex, bool is_double) {
    const int32_t forced = settings().geometry.latrd_grid_min_n;   // BATCHLAS_LATRD_GRID_MIN_N
    return forced > 0 ? forced : latrd_grid_min_n_default(cuda_cc, is_complex, is_double);
}

} // namespace batchlas::syev_tuning

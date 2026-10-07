// Helpers shared by syev_two_stage and syevx_direct_subset. Both run the Householder
// chase in both jobz modes at a real band width kd: choose_two_stage_kd_for_job is not
// safe for a path that never applies stage-2 reflectors.
// build_phase_from_kd1_band / apply_phase_rows act on a kd = 1 band (sb2st_hh's
// tridiagonal output in syev_two_stage, the sy2sb output in syevx_direct_subset).

#pragma once

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>

#include "../queue.hh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <batchlas/settings.hh>

namespace batchlas::two_stage_detail {

// Both knobs below: a value that does not parse, or parses to <= 0, means unset
// (env_positive_int_or, applied in settings.cc with the same literal defaults).

inline int32_t choose_two_stage_kd(int32_t n) {
    // kd = 32 is the measured optimum at every n >= 256, eigenvectors included; wide kd
    // does not win after the split-WY fix. Two-stage beats blocked only where the batch
    // saturates the device -- do not restate it as a plain "n >= 1024" rule.
    // evidence: docs/perf/syev.md#syev-the-two-stage-band-width-kd
    const int32_t kd = batchlas::settings().geometry.syev_two_stage_kd;  // default 32
    return std::min(std::max<int32_t>(1, kd), std::max<int32_t>(1, n - 1));
}

// Values-mode stage-2 chase: Householder by default (Givens is ~5x slower here);
// BATCHLAS_SYEV_TWO_STAGE_CHASE=givens restores Givens (values only). Read in the solve
// AND its *_buffer_size query: never make it stateful between the two.
// evidence: docs/perf/sytrd.md#sytrd-the-householder-chase-against-the-givens-chase
inline bool two_stage_use_givens_chase_for_values() {
    const char* v = batchlas::settings().selection.syev_two_stage_chase.get();
    return v && (std::string_view(v) == "givens");
}

inline int32_t choose_two_stage_kd_for_job(int32_t n, JobType jobz) {
    // Shared by both modes. The literature puts the eigenvector optimum higher; the
    // measurement here did not, so a per-mode split is a tuning question only.
    // evidence: docs/perf/syev.md#syev-the-two-stage-band-width-kd
    (void)jobz;
    return choose_two_stage_kd(n);
}

inline int32_t choose_two_stage_sb2st_block_size() {
    return batchlas::settings().geometry.syev_two_stage_sb2st_block;  // default 32
}

template <typename T>
inline void build_phase_from_kd1_band(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& ab_kd1,
                                       const VectorView<T>& phase) {
    using Real = typename base_type<T>::type;
    const int32_t n = static_cast<int32_t>(ab_kd1.cols());
    const int32_t batch = static_cast<int32_t>(ab_kd1.batch_size());
    if (n <= 0) return;

    ctx->submit([&](sycl::handler& cgh) {
        auto AB = ab_kd1.kernel_view();
        auto P = phase;
        cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(batch)), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);
            P(0, b) = T(1);
            for (int32_t i = 0; i < n - 1; ++i) {
                const T t = AB(1, i, b);
                Real a = Real(0);
                if constexpr (is_std_complex_v<T>) {
                    a = sycl::hypot(static_cast<Real>(t.real()), static_cast<Real>(t.imag()));
                } else {
                    a = sycl::fabs(t);
                }
                if (a == Real(0)) {
                    P(i + 1, b) = P(i, b);
                } else {
                    P(i + 1, b) = P(i, b) * (t / T(a));
                }
            }
        });
    });
}

// Scales row i of z by phase(i). z need not be square: the subset path applies
// this to an n x k block, so the column count must come from cols(), not rows().
template <typename T>
inline void apply_phase_rows(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& z,
                             const VectorView<T>& phase) {
    const int32_t n = static_cast<int32_t>(z.rows());
    const int32_t m = static_cast<int32_t>(z.cols());
    const int32_t batch = static_cast<int32_t>(z.batch_size());
    const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(n) * static_cast<int64_t>(m);
    if (total <= 0) return;
    ctx->submit([&](sycl::handler& cgh) {
        auto Z = z.kernel_view();
        auto P = phase;
        cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(n) * m));
            const int64_t rem = idx - static_cast<int64_t>(b) * n * m;
            const int32_t row = static_cast<int32_t>(rem % n);
            const int32_t col = static_cast<int32_t>(rem / n);
            Z(row, col, b) *= P(row, b);
        });
    });
}

// Applies the per-row phase to a real eigenvector block, writing T: complex T lifts the
// real column to (x, 0) before scaling; for real T the phase is a sign.
template <typename T>
inline void lift_eigvecs_with_phase(Queue& ctx,
                                    const MatrixView<typename base_type<T>::type, MatrixFormat::Dense>& z_real,
                                    const VectorView<T>& phase,
                                    const MatrixView<T, MatrixFormat::Dense>& z_out) {
    using Real = typename base_type<T>::type;
    const int32_t n = static_cast<int32_t>(z_real.rows());
    const int32_t batch = static_cast<int32_t>(z_real.batch_size());
    const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(n) * static_cast<int64_t>(n);
    ctx->submit([&](sycl::handler& cgh) {
        auto Zr = z_real.kernel_view();
        auto Zo = z_out.kernel_view();
        auto P = phase;
        cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(n) * n));
            const int64_t rem = idx - static_cast<int64_t>(b) * n * n;
            const int32_t row = static_cast<int32_t>(rem % n);
            const int32_t col = static_cast<int32_t>(rem / n);
            if constexpr (is_std_complex_v<T>) {
                Zo(row, col, b) = P(row, b) * T(Zr(row, col, b), Real(0));
            } else {
                Zo(row, col, b) = P(row, b) * Zr(row, col, b);
            }
        });
    });
}


} // namespace batchlas::two_stage_detail

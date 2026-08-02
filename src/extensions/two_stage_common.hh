// Helpers shared by the two-stage reduction paths (syev_two_stage, syevx_direct_subset).
//
// NOTE on kd: syev_two_stage's eigenvector path no longer forces kd = 1 -- since
// sytrd_sb2st_hh retains Q2, it reduces at a real band width and then applies the
// stage-2 reflectors explicitly. syevx_direct_subset has NOT been ported to that
// path: it still uses the Givens sytrd_sb2st, which discards Q2, so it pins kd = 1
// locally to make the band stage a pure extract. Do not assume the value returned by
// choose_two_stage_kd_for_job is safe for a path that never applies stage-2
// reflectors.
//
// build_phase_from_kd1_band and apply_phase_rows operate on a kd = 1 band regardless
// of the reduction width -- in syev_two_stage that band is sb2st_hh's tridiagonal
// output, in syevx_direct_subset it is the sy2sb output directly.

#pragma once

#include <blas/extensions.hh>
#include <blas/linalg.hh>
#include <blas/matrix.hh>

#include "../queue.hh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace batchlas::two_stage_detail {

inline int32_t env_int_or_default(const char* key, int32_t defval) {
    const char* v = std::getenv(key);
    if (!v || !*v) return defval;
    const int parsed = std::atoi(v);
    return (parsed > 0) ? static_cast<int32_t>(parsed) : defval;
}

inline int32_t choose_two_stage_kd(int32_t n) {
    // Measured with syev_two_stage_benchmark (float, eigenvectors, RTX 4090,
    // total ms; kd across, n/batch down):
    //
    //                kd=16      32      48      64      96   blocked
    //   128/2048      27.8    23.4    22.8    21.9    19.6     15.0
    //   256/1024      78.9    65.3    66.9    66.7    72.2     42.4
    //   512/512      249.5   203.9   223.1   240.0   298.6    193.3
    //   1024/128     500.1   425.8   443.9   470.0   546.6    481.4
    //   2048/32     1275.3  1183.4  1259.0  1353.8  1614.0   1265.7
    //
    // kd=32 is optimal at every n >= 256. This supersedes an earlier 32/64 split
    // measured before the wave back-transform landed: back then Q2 dominated and
    // its cost fell with kd, which pulled the optimum up to 64 at large n. Now
    // that Q2 is ~3x cheaper the balance is set by stage 1 and the chase, whose
    // O(n^2 kd) work favours a narrow band, so the optimum came back down.
    //
    // Two-stage now *wins* at n >= 1024 (1.13x at n=1024, 1.06x at n=2048) and
    // still loses below that, where blocked's lower fixed overhead dominates.
    const int32_t def = 32;

    const int32_t kd = env_int_or_default("BATCHLAS_SYEV_TWO_STAGE_KD", def);
    return std::min(std::max<int32_t>(1, kd), std::max<int32_t>(1, n - 1));
}

inline int32_t choose_two_stage_kd_for_job(int32_t n, JobType jobz) {
    // Eigenvector mode used to force kd=1 because the Givens stage-2 discards
    // Q2. sytrd_sb2st_hh retains it, so both modes now use a real band width.
    //
    // Note the tuning literature has the optimum going *up*, not down, when
    // eigenvectors are wanted (Gates/Tomov/Dongarra 2018 measure GPU 32/64
    // without vectors -> 96/128 with; MAGMA's get_nb.cpp uses band nb=128),
    // because the extra back-transform favours large nb while only stage 2's
    // O(n^2 nb) work favours small. choose_two_stage_kd is left shared for now;
    // splitting it is a tuning question, not a correctness one.
    (void)jobz;
    return choose_two_stage_kd(n);
}

inline int32_t choose_two_stage_sb2st_block_size() {
    return env_int_or_default("BATCHLAS_SYEV_TWO_STAGE_SB2ST_BLOCK", 32);
}

template <typename T>
inline void pack_sytrd_lower_to_qsub_qr_layout(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& a_sytrd,
                                                const MatrixView<T, MatrixFormat::Dense>& a_qsub_qr,
                                                const VectorView<T>& tau_sytrd,
                                                const VectorView<T>& tau_qsub,
                                                int32_t n) {
    const int32_t batch = static_cast<int32_t>(a_sytrd.batch_size());
    const int32_t p = std::max<int32_t>(0, n - 1);
    if (p == 0) return;

    ctx->submit([&](sycl::handler& cgh) {
        auto A = a_sytrd.kernel_view();
        auto AQ = a_qsub_qr.kernel_view();
        const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(p) * static_cast<int64_t>(p);
        cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(p) * p));
            const int64_t rem = idx - static_cast<int64_t>(b) * p * p;
            const int32_t row = static_cast<int32_t>(rem % p);
            const int32_t col = static_cast<int32_t>(rem / p);

            T val = T(0);
            if (row > col) {
                val = A(row + 1, col, b);
            }
            AQ(row, col, b) = val;
        });
    });

    ctx->submit([&](sycl::handler& cgh) {
        auto TAU = tau_sytrd;
        auto TAUQ = tau_qsub;
        const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(p);
        cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / p);
            const int32_t i = static_cast<int32_t>(idx - static_cast<int64_t>(b) * p);
            TAUQ(i, b) = TAU(i, b);
        });
    });
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

// Same as lift_real_eigvecs_with_phase but also valid for real T, where the
// phase is just a sign. Used by the eigenvector path, which is now shared
// between the real and complex cases.
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

template <typename T>
inline void lift_real_eigvecs_with_phase(Queue& ctx,
                                         const MatrixView<typename base_type<T>::type, MatrixFormat::Dense>& z_real,
                                         const VectorView<T>& phase,
                                         const MatrixView<T, MatrixFormat::Dense>& z_complex) {
    using Real = typename base_type<T>::type;
    const int32_t n = static_cast<int32_t>(z_real.rows());
    const int32_t batch = static_cast<int32_t>(z_real.batch_size());
    const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(n) * static_cast<int64_t>(n);
    ctx->submit([&](sycl::handler& cgh) {
        auto Zr = z_real.kernel_view();
        auto Zc = z_complex.kernel_view();
        auto P = phase;
        cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(n) * n));
            const int64_t rem = idx - static_cast<int64_t>(b) * n * n;
            const int32_t row = static_cast<int32_t>(rem % n);
            const int32_t col = static_cast<int32_t>(rem / n);
            Zc(row, col, b) = P(row, b) * T(Zr(row, col, b), Real(0));
        });
    });
}


} // namespace batchlas::two_stage_detail

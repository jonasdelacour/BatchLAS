#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/linalg.hh>
#include <blas/matrix.hh>
#include <internal/ormqr_blocked.hh>
#include <util/mempool.hh>

#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include "../queue.hh"
#include "../util/template-instantiations.hh"
#include "sytrd_sb2st_hh.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace batchlas {

namespace {

template <typename T>
inline void validate_syev_two_stage_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                         Span<typename base_type<T>::type> eigenvalues,
                                         JobType jobz,
                                         Uplo uplo) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("syev_two_stage: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw std::invalid_argument("syev_two_stage: invalid JobType.");
    }
    if (uplo != Uplo::Lower) {
        throw std::invalid_argument("syev_two_stage: only Uplo::Lower is currently implemented.");
    }

    const int64_t n64 = a.rows();
    const int64_t batch64 = a.batch_size();
    if (n64 < 1 || batch64 < 1) {
        throw std::invalid_argument("syev_two_stage: invalid n or batch size.");
    }

    const std::size_t need = static_cast<std::size_t>(n64) * static_cast<std::size_t>(batch64);
    if (eigenvalues.size() < need) {
        throw std::invalid_argument("syev_two_stage: eigenvalues span too small for n*batch.");
    }
}

inline int32_t env_int_or_default(const char* key, int32_t defval) {
    const char* v = std::getenv(key);
    if (!v || !*v) return defval;
    const int parsed = std::atoi(v);
    return (parsed > 0) ? static_cast<int32_t>(parsed) : defval;
}

// sytrd_sy2sb produces a numerically wrong band unless the trailing panel is
// degenerate, i.e. unless n % kd is 0 or 1. Verified exhaustively for float over
// kd in {16,24,32,48,64,96} x n in {64,96,128,129,192,256}: every cell with
// n % kd > 1 fails the eigenvector residual at O(0.1..0.8), every cell with
// n % kd <= 1 passes, with no other pattern (it is not "kd > n/2", and it is not
// complex-only as the earlier fallback comment below assumed -- kd=16 and kd=32
// looked clean only because every n tested against them happened to satisfy the
// rule). SyevTwoStageTest.Sy2sbTrailingPanelIsWrong pins this down.
//
// Until stage 1 is fixed, kd selection must respect the rule.
constexpr int32_t kMinTwoStageKd = 8;

inline bool sy2sb_kd_is_safe(int32_t n, int32_t kd) {
    return kd <= 1 || (n % kd) <= 1;
}

inline int32_t choose_two_stage_kd(int32_t n) {
    // Measured with syev_two_stage_benchmark (float, eigenvectors, RTX 4090,
    // total ms; kd across, n/batch down). Only kd values satisfying
    // sy2sb_kd_is_safe are listed: kd=48 and kd=96 time ~5-10% better at some
    // sizes but violate the rule at every n here, so those numbers are garbage
    // runs, not options. That trap is why this table is restricted by hand.
    //
    //                kd=16      32      64    blocked
    //   128/2048      28.4    23.4    21.1      15.0
    //   256/1024      80.4    67.6    68.1      41.8
    //   512/512      328.3   254.4   263.5     193.4
    //   1024/128     987.9   641.3   555.1     479.9
    //   2048/32     3201.8  2004.5  1711.2    1265.2
    //
    // The optimum rises with n (Gates/Tomov/Dongarra 2018 report the same trend
    // once eigenvectors are wanted), so this is a table rather than a constant.
    // The old flat 16/32 rule was off by 1.16x at n=1024 and 1.17x at n=2048.
    // n=128 also prefers 64, but two-stage loses to blocked by 1.4x there, so
    // the simpler threshold is kept rather than special-casing an unused size.
    //
    // Note the last column: blocked still wins everywhere, by 1.16x at n=1024
    // (its best case for two-stage) and more elsewhere.
    const int32_t def = (n <= 512) ? 32 : 64;

    // An explicit override is taken verbatim (it is how the kd sweep in
    // syev_two_stage_benchmark works); it is the caller's job to respect
    // sy2sb_kd_is_safe.
    if (const char* ev = std::getenv("BATCHLAS_SYEV_TWO_STAGE_KD")) {
        const int32_t kd = static_cast<int32_t>(std::atol(ev));
        if (kd > 0) return std::min(kd, std::max<int32_t>(1, n - 1));
    }

    const int32_t target = std::min(def, std::max<int32_t>(1, n - 1));
    for (int32_t kd = target; kd >= kMinTwoStageKd; --kd) {
        if (sy2sb_kd_is_safe(n, kd)) return kd;
    }
    return 0;  // caller falls back to syev_blocked
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

template <typename T>
inline void apply_phase_rows(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& z,
                             const VectorView<T>& phase) {
    const int32_t n = static_cast<int32_t>(z.rows());
    const int32_t batch = static_cast<int32_t>(z.batch_size());
    const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(n) * static_cast<int64_t>(n);
    ctx->submit([&](sycl::handler& cgh) {
        auto Z = z.kernel_view();
        auto P = phase;
        cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(n) * n));
            const int64_t rem = idx - static_cast<int64_t>(b) * n * n;
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

} // namespace

template <Backend B, typename T>
Event syev_two_stage(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& a_in,
                     Span<typename base_type<T>::type> eigenvalues,
                     JobType jobz,
                     Uplo uplo,
                     const Span<std::byte>& ws,
                     StedcParams<typename base_type<T>::type> stedc_params) {
    BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.entry");
    validate_syev_two_stage_dims(a_in, eigenvalues, jobz, uplo);

    if (!ctx.in_order()) {
        throw std::runtime_error("syev_two_stage: requires an in-order Queue");
    }

    if constexpr (B == Backend::NETLIB) {
        if (jobz == JobType::EigenVectors) {
            return syev_blocked<B, T>(ctx, a_in, eigenvalues, jobz, uplo, ws, stedc_params);
        }
    }

    if constexpr (is_std_complex_v<T>) {
        if (jobz == JobType::EigenVectors) {
            // sytrd_sy2sb is not accurate for complex input when n is not a
            // multiple of kd: an isolated spectrum-preservation check on stage 1
            // alone (SyevTwoStageTest.Sy2sbBandPreservesSpectrum) is clean for
            // float and double but fails for complex<float> (~8e-2) and
            // complex<double> (~1e-5) at n=129/kd=16, while n=128 passes.
            // sytrd_sy2sb_tests covers only float and double, so this was never
            // caught; eigenvector mode is the first caller to use kd>1.
            //
            // Falling back keeps complex results correct. Remove this once
            // stage 1's complex path is fixed.
            return syev_blocked<B, T>(ctx, a_in, eigenvalues, jobz, uplo, ws, stedc_params);
        }
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    const bool want_eigvecs = (jobz == JobType::EigenVectors);

    // No band width in the useful range gives sy2sb a degenerate trailing panel
    // (see sy2sb_kd_is_safe); running anyway would return silent garbage.
    if (choose_two_stage_kd_for_job(n, jobz) == 0) {
        return syev_blocked<B, T>(ctx, a_in, eigenvalues, jobz, uplo, ws, stedc_params);
    }
    const int32_t kd = choose_two_stage_kd_for_job(n, jobz);
    const int32_t tau_sy2sb_n = std::max<int32_t>(0, n - kd);
    const int32_t sb2st_block_size = choose_two_stage_sb2st_block_size();
    const int32_t p = std::max<int32_t>(0, n - 1);
    const int32_t ormqr_block_size = tuning::ormqr_block_size_for_n(n);

    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    // Stage 1 workspace and outputs: dense -> band.
    auto ab_span = pool.allocate<T>(ctx,
                                    static_cast<std::size_t>(kd + 1) *
                                        static_cast<std::size_t>(n) *
                                        static_cast<std::size_t>(batch));
    MatrixView<T, MatrixFormat::Dense> ab_view(ab_span.data(),
                                               kd + 1,
                                               n,
                                               kd + 1,
                                               static_cast<int64_t>(kd + 1) * static_cast<int64_t>(n),
                                               batch);

    auto tau_sy2sb_span = pool.allocate<T>(ctx,
                                           static_cast<std::size_t>(tau_sy2sb_n) *
                                               static_cast<std::size_t>(batch));
    VectorView<T> tau_sy2sb_view(tau_sy2sb_span,
                                 tau_sy2sb_n,
                                 batch,
                                 1,
                                 tau_sy2sb_n);

    Span<T> phase_span;
    VectorView<T> phase_view;
    if (want_eigvecs) {
        phase_span = pool.allocate<T>(ctx,
                                      static_cast<std::size_t>(n) *
                                          static_cast<std::size_t>(batch));
        phase_view = VectorView<T>(phase_span, n, batch, 1, n);
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.sy2sb");
        const size_t sy2sb_ws_bytes = sytrd_sy2sb_buffer_size<B, T>(ctx,
                                                                     a,
                                                                     ab_view,
                                                                     tau_sy2sb_view,
                                                                     uplo,
                                                                     kd);
        auto sy2sb_ws = pool.allocate<std::byte>(ctx, sy2sb_ws_bytes);
        sytrd_sy2sb<B, T>(ctx, a, ab_view, tau_sy2sb_view, uplo, kd, sy2sb_ws);
    }

    // Stage 2 outputs: band -> tridiagonal (real d,e).
    using Real = typename base_type<T>::type;
    auto d_span = pool.allocate<Real>(ctx,
                                      static_cast<std::size_t>(n) *
                                          static_cast<std::size_t>(batch));
    auto e_span = pool.allocate<Real>(ctx,
                                      static_cast<std::size_t>(std::max(0, n - 1)) *
                                          static_cast<std::size_t>(batch));
    auto tau_sb2st_span = pool.allocate<T>(ctx,
                                           static_cast<std::size_t>(std::max(0, n - 1)) *
                                               static_cast<std::size_t>(batch));

    VectorView<Real> d_view(d_span, n, batch, 1, n);
    VectorView<Real> e_view(e_span, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
    VectorView<T> tau_sb2st_view(tau_sb2st_span,
                                 std::max(0, n - 1),
                                 batch,
                                 1,
                                 std::max(0, n - 1));

    // Stage 2. Eigenvector mode uses the Householder chase so that Q2 is
    // retained; eigenvalues-only keeps the cheaper Givens chase, which discards
    // it. This is the whole reason kd no longer has to be clamped to 1.
    const auto sb2st_sched = want_eigvecs ? internal::build_sb2st_hh_schedule(n, kd)
                                          : std::vector<internal::Sb2stHhRefl>{};
    const int32_t nrefl = static_cast<int32_t>(sb2st_sched.size());

    UnifiedVector<int32_t> sb2st_starts(static_cast<std::size_t>(nrefl));
    UnifiedVector<int32_t> sb2st_lens(static_cast<std::size_t>(nrefl));
    for (int32_t k = 0; k < nrefl; ++k) {
        sb2st_starts[k] = sb2st_sched[k].start;
        sb2st_lens[k] = sb2st_sched[k].len;
    }

    Span<T> v_sb2st_span;
    Span<T> tau_sb2st_hh_span;
    Span<T> ab_tri_span;
    MatrixView<T, MatrixFormat::Dense> v_sb2st_view;
    VectorView<T> tau_sb2st_hh_view;
    MatrixView<T, MatrixFormat::Dense> ab_tri_view;

    if (want_eigvecs) {
        const int32_t nr = std::max<int32_t>(1, nrefl);
        v_sb2st_span = pool.allocate<T>(ctx,
                                        static_cast<std::size_t>(kd) *
                                            static_cast<std::size_t>(nr) *
                                            static_cast<std::size_t>(batch));
        tau_sb2st_hh_span = pool.allocate<T>(ctx,
                                             static_cast<std::size_t>(nr) *
                                                 static_cast<std::size_t>(batch));
        ab_tri_span = pool.allocate<T>(ctx,
                                       static_cast<std::size_t>(2) *
                                           static_cast<std::size_t>(n) *
                                           static_cast<std::size_t>(batch));
        v_sb2st_view = MatrixView<T, MatrixFormat::Dense>(
            v_sb2st_span.data(), kd, nr, kd,
            static_cast<int64_t>(kd) * static_cast<int64_t>(nr), batch);
        tau_sb2st_hh_view = VectorView<T>(tau_sb2st_hh_span, nr, batch, 1, nr);
        ab_tri_view = MatrixView<T, MatrixFormat::Dense>(
            ab_tri_span.data(), 2, n, 2, static_cast<int64_t>(2) * static_cast<int64_t>(n), batch);

        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.sb2st_hh");
        const size_t ws_bytes = internal::sytrd_sb2st_hh_buffer_size<B, T>(ctx, n, kd, batch);
        auto hh_ws = pool.allocate<std::byte>(ctx, ws_bytes);
        internal::sytrd_sb2st_hh<B, T>(ctx, ab_view, ab_tri_view, d_view, e_view,
                                       v_sb2st_view, tau_sb2st_hh_view, uplo, kd,
                                       hh_ws);

        build_phase_from_kd1_band<T>(ctx, ab_tri_view, phase_view);
    } else {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.sb2st");
        const size_t sb2st_ws_bytes = sytrd_sb2st_buffer_size<B, T>(ctx,
                                                                     ab_view,
                                                                     d_view,
                                                                     e_view,
                                                                     tau_sb2st_view,
                                                                     uplo,
                                                                     kd,
                                                                     sb2st_block_size);
        auto sb2st_ws = pool.allocate<std::byte>(ctx, sb2st_ws_bytes);
        sytrd_sb2st<B, T>(ctx,
                          ab_view,
                          d_view,
                          e_view,
                          tau_sb2st_view,
                          uplo,
                          kd,
                          sb2st_ws,
                          sb2st_block_size);
    }

    VectorView<Real> evals_view(eigenvalues.data(), n, batch, 1, n);

    if (!want_eigvecs) {
        auto z_span = pool.allocate<Real>(ctx,
                                          static_cast<std::size_t>(n) *
                                              static_cast<std::size_t>(n) *
                                              static_cast<std::size_t>(batch));
        MatrixView<Real, MatrixFormat::Dense> z_view(z_span.data(),
                                                     n,
                                                     n,
                                                     n,
                                                     static_cast<int64_t>(n) * static_cast<int64_t>(n),
                                                     batch);

        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.stedc_evals");
        const size_t stedc_ws_bytes = stedc_workspace_size<B, Real>(ctx,
                                                                     static_cast<std::size_t>(n),
                                                                     static_cast<std::size_t>(batch),
                                                                     JobType::NoEigenVectors,
                                                                     stedc_params);
        auto stedc_ws = pool.allocate<std::byte>(ctx, stedc_ws_bytes);
        stedc<B, Real>(ctx,
                       d_view,
                       e_view,
                       evals_view,
                       stedc_ws,
                       JobType::NoEigenVectors,
                       stedc_params,
                       z_view);

        return ctx.get_event();
    }

    // ---- Eigenvector path -------------------------------------------------
    //
    // Stage 2 ran the Householder chase, so Q2 exists and kd was NOT clamped to
    // 1. The back-transform is Z := Q1 (Q2 Z). That ordering costs ~4n^3;
    // forming (Q1 Q2) explicitly first would be ~5.3n^3 (it needs Q1
    // materialised) and is only worth it if the extra work can be overlapped
    // with stedc, which the in-order queue does not currently allow.
    const int32_t p1 = std::max<int32_t>(0, n - kd);

    auto z_real_span = pool.allocate<Real>(ctx,
                                           static_cast<std::size_t>(n) *
                                               static_cast<std::size_t>(n) *
                                               static_cast<std::size_t>(batch));
    MatrixView<Real, MatrixFormat::Dense> z_real_view(z_real_span.data(),
                                                      n,
                                                      n,
                                                      n,
                                                      static_cast<int64_t>(n) * static_cast<int64_t>(n),
                                                      batch);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.stedc_eigvecs");
        const size_t stedc_ws_bytes = stedc_workspace_size<B, Real>(ctx,
                                                                     static_cast<std::size_t>(n),
                                                                     static_cast<std::size_t>(batch),
                                                                     JobType::EigenVectors,
                                                                     stedc_params);
        auto stedc_ws = pool.allocate<std::byte>(ctx, stedc_ws_bytes);
        stedc<B, Real>(ctx,
                       d_view,
                       e_view,
                       evals_view,
                       stedc_ws,
                       JobType::EigenVectors,
                       stedc_params,
                       z_real_view);
    }

    // stedc solves the *real* tridiagonal built from |subdiagonal|. Lifting by
    // the accumulated phase converts its eigenvectors back to those of the
    // signed/Hermitian tridiagonal that stage 2 actually produced.
    auto z_span = pool.allocate<T>(ctx,
                                   static_cast<std::size_t>(n) *
                                       static_cast<std::size_t>(n) *
                                       static_cast<std::size_t>(batch));
    MatrixView<T, MatrixFormat::Dense> z_view(z_span.data(),
                                              n,
                                              n,
                                              n,
                                              static_cast<int64_t>(n) * static_cast<int64_t>(n),
                                              batch);
    lift_eigvecs_with_phase<T>(ctx, z_real_view, phase_view, z_view);

    // Z := Q2 Z
    if (nrefl > 0) {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.backtransform_q2");
        internal::unmqr_hb2st<B, T>(ctx,
                                    v_sb2st_view,
                                    tau_sb2st_hh_view,
                                    z_view,
                                    n,
                                    kd,
                                    Span<const int32_t>(sb2st_starts.data(), sb2st_starts.size()),
                                    Span<const int32_t>(sb2st_lens.data(), sb2st_lens.size()));
    }

    // Z(kd:, :) := Q1 Z(kd:, :).
    //
    // sy2sb factors panel i with GEQRF starting at row i+kd, so the aggregate
    // a(kd:, 0:n-kd) is already exactly a GEQRF-style reflector layout: panel i
    // sits at local (row i, col i). ormqr_blocked's V packing only reads the
    // strictly lower triangle, so the sliced view suffices -- no packed copy,
    // matching what syev_blocked does (sytrd_lower_qsub_reflector_view).
    if (p1 > 0) {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.backtransform_q1");
        auto v1_view = a({kd, SliceEnd()}, {0, p1});
        auto z_sub = z_view({kd, SliceEnd()}, Slice{});
        Span<T> tau1_flat(tau_sy2sb_span.data(),
                          static_cast<std::size_t>(p1) * static_cast<std::size_t>(batch));

        size_t ormqr_ws_bytes = 0;
        if constexpr (B == Backend::NETLIB) {
            ormqr_ws_bytes = backend::ormqr_vendor_buffer_size<B, T>(
                ctx, v1_view, z_sub, Side::Left, Transpose::NoTrans, tau1_flat);
        } else {
            ormqr_ws_bytes = ormqr_blocked_buffer_size<B, T>(
                ctx, v1_view, z_sub, Side::Left, Transpose::NoTrans, tau1_flat, ormqr_block_size);
        }
        auto ormqr_ws = pool.allocate<std::byte>(ctx, ormqr_ws_bytes);
        if constexpr (B == Backend::NETLIB) {
            backend::ormqr_vendor<B, T>(
                ctx, v1_view, z_sub, Side::Left, Transpose::NoTrans, tau1_flat, ormqr_ws);
        } else {
            ormqr_blocked<B, T>(ctx, v1_view, z_sub, Side::Left, Transpose::NoTrans,
                                tau1_flat, ormqr_ws, ormqr_block_size);
        }
    }

    MatrixView<T, MatrixFormat::Dense>::copy(ctx, a, z_view);

    return ctx.get_event();
}

template <Backend B, typename T>
size_t syev_two_stage_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& a,
                                  JobType jobz,
                                  Uplo uplo,
                                  StedcParams<typename base_type<T>::type> stedc_params) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("syev_two_stage_buffer_size: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw std::invalid_argument("syev_two_stage_buffer_size: invalid JobType.");
    }
    if (uplo != Uplo::Lower) {
        throw std::invalid_argument("syev_two_stage_buffer_size: only Uplo::Lower is currently implemented.");
    }

    if constexpr (B == Backend::NETLIB) {
        if (jobz == JobType::EigenVectors) {
            return syev_blocked_buffer_size<B, T>(ctx, a, jobz, uplo, stedc_params);
        }
    }

    if constexpr (is_std_complex_v<T>) {
        if (jobz == JobType::EigenVectors) {
            // Mirrors the fallback in syev_two_stage (stage-1 complex accuracy).
            return syev_blocked_buffer_size<B, T>(ctx, a, jobz, uplo, stedc_params);
        }
    }

    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    const bool want_eigvecs = (jobz == JobType::EigenVectors);
    const int32_t kd = choose_two_stage_kd_for_job(n, jobz);

    // Must mirror the fallback in syev_two_stage exactly, or the workspace and
    // the path that consumes it disagree.
    if (kd == 0) {
        return syev_blocked_buffer_size<B, T>(ctx, a, jobz, uplo, stedc_params);
    }

    const int32_t tau_sy2sb_n = std::max<int32_t>(0, n - kd);
    const int32_t sb2st_block_size = choose_two_stage_sb2st_block_size();
    const int32_t p = std::max<int32_t>(0, n - 1);
    const int32_t ormqr_block_size = tuning::ormqr_block_size_for_n(n);

    using Real = typename base_type<T>::type;

    size_t bytes = 0;

    // Persistent buffers for the three stages.
    bytes += BumpAllocator::allocation_size<T>(ctx,
                                               static_cast<std::size_t>(kd + 1) *
                                                   static_cast<std::size_t>(n) *
                                                   static_cast<std::size_t>(batch)); // AB
    bytes += BumpAllocator::allocation_size<T>(ctx,
                                               static_cast<std::size_t>(tau_sy2sb_n) *
                                                   static_cast<std::size_t>(batch)); // tau sy2sb
    bytes += BumpAllocator::allocation_size<Real>(ctx,
                                                  static_cast<std::size_t>(n) *
                                                      static_cast<std::size_t>(batch)); // d
    bytes += BumpAllocator::allocation_size<Real>(ctx,
                                                  static_cast<std::size_t>(std::max(0, n - 1)) *
                                                      static_cast<std::size_t>(batch)); // e
    bytes += BumpAllocator::allocation_size<T>(ctx,
                                               static_cast<std::size_t>(std::max(0, n - 1)) *
                                                   static_cast<std::size_t>(batch)); // tau sb2st
    bytes += BumpAllocator::allocation_size<Real>(ctx,
                                                  static_cast<std::size_t>(n) *
                                                      static_cast<std::size_t>(n) *
                                                      static_cast<std::size_t>(batch)); // stedc scratch/result
    if (want_eigvecs) {
        const int32_t nr = std::max<int32_t>(1, internal::sb2st_hh_num_reflectors(n, kd));
        const int32_t kdw = internal::sb2st_hh_work_bandwidth(n, kd);
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(n) *
                                                       static_cast<std::size_t>(batch)); // phase/sign chain
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(kd) *
                                                       static_cast<std::size_t>(nr) *
                                                       static_cast<std::size_t>(batch)); // stage-2 reflectors V
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(nr) *
                                                       static_cast<std::size_t>(batch)); // stage-2 tau
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(2) *
                                                       static_cast<std::size_t>(n) *
                                                       static_cast<std::size_t>(batch)); // tridiagonal band (signed)
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(kdw + 1) *
                                                       static_cast<std::size_t>(n) *
                                                       static_cast<std::size_t>(batch)); // sb2st_hh working band
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(n) *
                                                       static_cast<std::size_t>(n) *
                                                       static_cast<std::size_t>(batch)); // Z in T (post phase-lift)
    }

    MatrixView<T, MatrixFormat::Dense> ab_dummy(nullptr,
                                                kd + 1,
                                                n,
                                                kd + 1,
                                                static_cast<int64_t>(kd + 1) * static_cast<int64_t>(n),
                                                batch);
    VectorView<T> tau_sy2sb_dummy(nullptr,
                                  tau_sy2sb_n,
                                  batch,
                                  1,
                                  tau_sy2sb_n);

    VectorView<Real> d_dummy(nullptr, n, batch, 1, n);
    VectorView<Real> e_dummy(nullptr, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
    VectorView<T> tau_sb2st_dummy(nullptr,
                                  std::max(0, n - 1),
                                  batch,
                                  1,
                                  std::max(0, n - 1));

    bytes += sytrd_sy2sb_buffer_size<B, T>(ctx, a, ab_dummy, tau_sy2sb_dummy, uplo, kd);
    bytes += sytrd_sb2st_buffer_size<B, T>(ctx,
                                           ab_dummy,
                                           d_dummy,
                                           e_dummy,
                                           tau_sb2st_dummy,
                                           uplo,
                                           kd,
                                           sb2st_block_size);
    bytes += stedc_workspace_size<B, Real>(ctx,
                                           static_cast<std::size_t>(n),
                                           static_cast<std::size_t>(batch),
                                           want_eigvecs ? JobType::EigenVectors : JobType::NoEigenVectors,
                                           stedc_params);

    if (want_eigvecs && p > 0) {
        MatrixView<T, MatrixFormat::Dense> aq_dummy(nullptr,
                                                    p,
                                                    p,
                                                    p,
                                                    static_cast<int64_t>(p) * static_cast<int64_t>(p),
                                                    batch);
        MatrixView<T, MatrixFormat::Dense> z_dummy(nullptr,
                                                   n,
                                                   n,
                                                   n,
                                                   static_cast<int64_t>(n) * static_cast<int64_t>(n),
                                                   batch);
        auto z_sub_dummy = z_dummy({1, SliceEnd()}, Slice{});
        Span<T> tau_q_dummy(nullptr, static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));
        if constexpr (B == Backend::NETLIB) {
            bytes += backend::ormqr_vendor_buffer_size<B, T>(ctx,
                                                              aq_dummy,
                                                              z_sub_dummy,
                                                              Side::Left,
                                                              Transpose::NoTrans,
                                                              tau_q_dummy);
        } else {
            bytes += ormqr_blocked_buffer_size<B, T>(ctx,
                                                     aq_dummy,
                                                     z_sub_dummy,
                                                     Side::Left,
                                                     Transpose::NoTrans,
                                                     tau_q_dummy,
                                                     ormqr_block_size);
        }
    }

    return bytes;
}

#define SYEV_TWO_STAGE_INSTANTIATE(back, fp) \
    template Event syev_two_stage<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        JobType, \
        Uplo, \
        const Span<std::byte>&, \
        StedcParams<typename base_type<BATCHLAS_UNPAREN fp>::type>); \
    template size_t syev_two_stage_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        JobType, \
        Uplo, \
        StedcParams<typename base_type<BATCHLAS_UNPAREN fp>::type>);

#define SYEV_TWO_STAGE_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEV_TWO_STAGE_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
SYEV_TWO_STAGE_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYEV_TWO_STAGE_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYEV_TWO_STAGE_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SYEV_TWO_STAGE_INSTANTIATE_FOR_BACKEND
#undef SYEV_TWO_STAGE_INSTANTIATE

} // namespace batchlas

#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/linalg.hh>
#include <blas/matrix.hh>
#include <internal/ormqr_blocked.hh>
#include <util/mempool.hh>

#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include "../queue.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace {

template <typename U>
struct is_std_complex : std::false_type {};

template <typename U>
struct is_std_complex<std::complex<U>> : std::true_type {};

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

inline int32_t choose_two_stage_kd(int32_t n) {
    // Conservative defaults for initial two-stage path; override with env as needed.
    const int32_t def = (n <= 256) ? 16 : 32;
    const int32_t kd = env_int_or_default("BATCHLAS_SYEV_TWO_STAGE_KD", def);
    return std::min(std::max<int32_t>(1, kd), std::max<int32_t>(1, n - 1));
}

inline int32_t choose_two_stage_kd_for_job(int32_t n, JobType jobz) {
    // Eigenvector mode currently requires kd=1 so stage-2 is a pure extract,
    // then Q from stage-1 reflectors can be applied explicitly.
    if (jobz == JobType::EigenVectors) {
        return 1;
    }
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
                if constexpr (is_std_complex<T>::value) {
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

    if constexpr (B == Backend::CUDA) {
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            if (jobz == JobType::EigenVectors) {
                // Temporary guard: CUDA zhe eigvec path in two-stage currently
                // exhibits a backend runtime crash for some sizes.
                return syev_blocked<B, T>(ctx, a_in, eigenvalues, jobz, uplo, ws, stedc_params);
            }
        }
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    const bool want_eigvecs = (jobz == JobType::EigenVectors);
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

    if (want_eigvecs) {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.phase_from_band");
        build_phase_from_kd1_band<T>(ctx, ab_view, phase_view);
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

    {
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

    auto aq_span = pool.allocate<T>(ctx,
                                    static_cast<std::size_t>(p) *
                                        static_cast<std::size_t>(p) *
                                        static_cast<std::size_t>(batch));
    auto tau_q_span = pool.allocate<T>(ctx,
                                       static_cast<std::size_t>(p) *
                                           static_cast<std::size_t>(batch));
    MatrixView<T, MatrixFormat::Dense> aq_view(aq_span.data(),
                                               p,
                                               p,
                                               p,
                                               static_cast<int64_t>(p) * static_cast<int64_t>(p),
                                               batch);
    VectorView<T> tau_q_view(tau_q_span, p, batch, 1, p);
    pack_sytrd_lower_to_qsub_qr_layout<T>(ctx, a, aq_view, tau_sy2sb_view, tau_q_view, n);

    Span<T> tau_q_flat(tau_q_span.data(), static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));

    if constexpr (is_std_complex<T>::value) {
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

        auto zc_span = pool.allocate<T>(ctx,
                                        static_cast<std::size_t>(n) *
                                            static_cast<std::size_t>(n) *
                                            static_cast<std::size_t>(batch));
        MatrixView<T, MatrixFormat::Dense> zc_view(zc_span.data(),
                                                   n,
                                                   n,
                                                   n,
                                                   static_cast<int64_t>(n) * static_cast<int64_t>(n),
                                                   batch);

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

        lift_real_eigvecs_with_phase<T>(ctx, z_real_view, phase_view, zc_view);

        if (p > 0) {
            BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.backtransform_q");
            auto zc_sub = zc_view({1, SliceEnd()}, Slice{});
            size_t ormqr_ws_bytes = 0;
            if constexpr (B == Backend::NETLIB) {
                ormqr_ws_bytes = backend::ormqr_vendor_buffer_size<B, T>(ctx,
                                                                          aq_view,
                                                                          zc_sub,
                                                                          Side::Left,
                                                                          Transpose::NoTrans,
                                                                          tau_q_flat);
            } else {
                ormqr_ws_bytes = ormqr_blocked_buffer_size<B, T>(ctx,
                                                                  aq_view,
                                                                  zc_sub,
                                                                  Side::Left,
                                                                  Transpose::NoTrans,
                                                                  tau_q_flat,
                                                                  ormqr_block_size);
            }
            auto ormqr_ws = pool.allocate<std::byte>(ctx, ormqr_ws_bytes);
            if constexpr (B == Backend::NETLIB) {
                backend::ormqr_vendor<B, T>(ctx,
                                            aq_view,
                                            zc_sub,
                                            Side::Left,
                                            Transpose::NoTrans,
                                            tau_q_flat,
                                            ormqr_ws);
            } else {
                ormqr_blocked<B, T>(ctx,
                                    aq_view,
                                    zc_sub,
                                    Side::Left,
                                    Transpose::NoTrans,
                                    tau_q_flat,
                                    ormqr_ws,
                                    ormqr_block_size);
            }
        }

        MatrixView<T, MatrixFormat::Dense>::copy(ctx, a, zc_view);
    } else {
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
                       z_view);

        apply_phase_rows<Real>(ctx, z_view, VectorView<Real>(phase_span.data(), n, batch, 1, n));

        if (p > 0) {
            BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.backtransform_q");
            auto z_sub = z_view({1, SliceEnd()}, Slice{});
            MatrixView<Real, MatrixFormat::Dense> aq_real_view(aq_span.data(),
                                                               p,
                                                               p,
                                                               p,
                                                               static_cast<int64_t>(p) * static_cast<int64_t>(p),
                                                               batch);
            Span<Real> tau_q_real(reinterpret_cast<Real*>(tau_q_span.data()),
                                  static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));

            size_t ormqr_ws_bytes = 0;
            if constexpr (B == Backend::NETLIB) {
                ormqr_ws_bytes = backend::ormqr_vendor_buffer_size<B, Real>(ctx,
                                                                             aq_real_view,
                                                                             z_sub,
                                                                             Side::Left,
                                                                             Transpose::NoTrans,
                                                                             tau_q_real);
            } else {
                ormqr_ws_bytes = ormqr_blocked_buffer_size<B, Real>(ctx,
                                                                     aq_real_view,
                                                                     z_sub,
                                                                     Side::Left,
                                                                     Transpose::NoTrans,
                                                                     tau_q_real,
                                                                     ormqr_block_size);
            }
            auto ormqr_ws = pool.allocate<std::byte>(ctx, ormqr_ws_bytes);
            if constexpr (B == Backend::NETLIB) {
                backend::ormqr_vendor<B, Real>(ctx,
                                               aq_real_view,
                                               z_sub,
                                               Side::Left,
                                               Transpose::NoTrans,
                                               tau_q_real,
                                               ormqr_ws);
            } else {
                ormqr_blocked<B, Real>(ctx,
                                       aq_real_view,
                                       z_sub,
                                       Side::Left,
                                       Transpose::NoTrans,
                                       tau_q_real,
                                       ormqr_ws,
                                       ormqr_block_size);
            }
        }

        MatrixView<Real, MatrixFormat::Dense>::copy(ctx,
                                                    MatrixView<Real, MatrixFormat::Dense>(a.data_ptr(), a.rows(), a.cols(), a.ld(), a.stride(), a.batch_size()),
                                                    z_view);
    }

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

    if constexpr (B == Backend::CUDA) {
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            if (jobz == JobType::EigenVectors) {
                return syev_blocked_buffer_size<B, T>(ctx, a, jobz, uplo, stedc_params);
            }
        }
    }

    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    const bool want_eigvecs = (jobz == JobType::EigenVectors);
    const int32_t kd = choose_two_stage_kd_for_job(n, jobz);
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
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(n) *
                                                       static_cast<std::size_t>(batch)); // phase/sign chain
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(p) *
                                                       static_cast<std::size_t>(p) *
                                                       static_cast<std::size_t>(batch)); // packed Qsub reflectors
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(p) *
                                                       static_cast<std::size_t>(batch)); // tau for packed Qsub
        if constexpr (is_std_complex<T>::value) {
            bytes += BumpAllocator::allocation_size<T>(ctx,
                                                       static_cast<std::size_t>(n) *
                                                           static_cast<std::size_t>(n) *
                                                           static_cast<std::size_t>(batch)); // lifted complex eigenvectors
        }
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
    template Event syev_two_stage<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo, \
        const Span<std::byte>&, \
        StedcParams<typename base_type<fp>::type>); \
    template size_t syev_two_stage_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        JobType, \
        Uplo, \
        StedcParams<typename base_type<fp>::type>);

#if BATCHLAS_HAS_CUDA_BACKEND
SYEV_TWO_STAGE_INSTANTIATE(Backend::CUDA, float)
SYEV_TWO_STAGE_INSTANTIATE(Backend::CUDA, double)
SYEV_TWO_STAGE_INSTANTIATE(Backend::CUDA, std::complex<float>)
SYEV_TWO_STAGE_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYEV_TWO_STAGE_INSTANTIATE(Backend::ROCM, float)
SYEV_TWO_STAGE_INSTANTIATE(Backend::ROCM, double)
SYEV_TWO_STAGE_INSTANTIATE(Backend::ROCM, std::complex<float>)
SYEV_TWO_STAGE_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYEV_TWO_STAGE_INSTANTIATE(Backend::NETLIB, float)
SYEV_TWO_STAGE_INSTANTIATE(Backend::NETLIB, double)
SYEV_TWO_STAGE_INSTANTIATE(Backend::NETLIB, std::complex<float>)
SYEV_TWO_STAGE_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef SYEV_TWO_STAGE_INSTANTIATE

} // namespace batchlas

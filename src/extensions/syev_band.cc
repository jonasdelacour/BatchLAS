// syev_band: symmetric/Hermitian eigensolver built on band reduction.
//
//   stage 1  sytrd_sy2sb           dense -> band (semibandwidth kd)
//   stage 2  sytrd_band_reduction  band  -> tridiagonal (BANDR1 bulge-chase)
//   stage 3  stedc                 tridiagonal eigensolve
//
// Eigenvalues only; see the contract in <blas/extensions.hh>.

#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/linalg.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>

#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include "../queue.hh"
#include "../util/template-instantiations.hh"

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
inline void validate_syev_band_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                    Span<typename base_type<T>::type> eigenvalues,
                                    JobType jobz,
                                    Uplo uplo) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("syev_band: A must be square.");
    }
    if (jobz == JobType::EigenVectors) {
        // Stage 2 discards the bulge-chasing reflectors, so the similarity
        // transform cannot be replayed onto the tridiagonal eigenvectors.
        throw std::invalid_argument(
            "syev_band: JobType::EigenVectors is not supported (band reduction is "
            "eigenvalues-only); use syev_blocked or syev_two_stage instead.");
    }
    if (jobz != JobType::NoEigenVectors) {
        throw std::invalid_argument("syev_band: invalid JobType.");
    }
    if (uplo != Uplo::Lower) {
        throw std::invalid_argument("syev_band: only Uplo::Lower is currently implemented.");
    }

    const int64_t n64 = a.rows();
    const int64_t batch64 = a.batch_size();
    if (n64 < 1 || batch64 < 1) {
        throw std::invalid_argument("syev_band: invalid n or batch size.");
    }

    const std::size_t need = static_cast<std::size_t>(n64) * static_cast<std::size_t>(batch64);
    if (eigenvalues.size() < need) {
        throw std::invalid_argument("syev_band: eigenvalues span too small for n*batch.");
    }
}

inline int32_t env_int_or(const char* key, int32_t defval) {
    const char* v = std::getenv(key);
    if (!v || !*v) return defval;
    const int parsed = std::atoi(v);
    return (parsed > 0) ? static_cast<int32_t>(parsed) : defval;
}

// Semibandwidth of the intermediate band form.
//
// Stage 1 is GEMM-rich and gets cheaper per unit of work as kd grows (wider
// panels, larger trailing updates). Stage 2's cost grows with kd. The chase
// step count is ~n^2/(nb*b) per sweep, so a kd that is too small starves
// stage 1 while a kd that is too large makes stage 2 dominate.
inline int32_t choose_syev_band_kd(int32_t n) {
    int32_t def;
    if (n <= 64) {
        def = 8;
    } else if (n <= 256) {
        def = 16;
    } else {
        def = 32;
    }
    const int32_t kd = env_int_or("BATCHLAS_SYEV_BAND_KD", def);
    return std::min(std::max<int32_t>(1, kd), std::max<int32_t>(1, n - 1));
}

// Diagonals removed per sweep.
//
// A sweep at bandwidth b costs ~n^2/(2*nb*b) chase steps, and the schedule
// constrains the panel width to nb <= b - d. So d trades sweep count against
// panel width, and the two do not balance: reducing all the way in one sweep
// (d = kd-1, forcing nb = 1) costs n^2/(2*kd) steps, while shaving one diagonal
// per sweep (d = 1, nb = kd-1) costs sum_b n^2/(2*(b-1)*b) ~ n^2/2 -- a factor
// of kd more. Any intermediate split is worse than the single sweep too, since
// the later, narrower sweeps dominate.
//
// Measured at n=256, kd=16, batch=1 (stage 2 alone): d=1 4969ms, d=2 3758ms,
// d=4 1553ms, d=8 686ms, d=15 327ms. Monotone, 15.2x end to end.
inline int32_t choose_syev_band_d(int32_t kd) {
    // kd is an upper bound; the implementation clamps d to b-1 per sweep.
    return env_int_or("BATCHLAS_SYEV_BAND_D", std::max<int32_t>(1, kd));
}

inline int32_t choose_syev_band_nb(int32_t n, int32_t kd) {
    const int32_t def = std::max<int32_t>(1, std::min<int32_t>(kd, 32));
    const int32_t nb = env_int_or("BATCHLAS_SYEV_BAND_NB", def);
    return std::max<int32_t>(1, nb);
}

inline SytrdBandReductionParams make_bandr_params(int32_t n, int32_t kd) {
    SytrdBandReductionParams p;
    p.d_seq = {choose_syev_band_d(kd)};
    // With d = kd-1 the schedule forces nb = b - d = 1 regardless; this only
    // matters when d is overridden downwards.
    p.block_size_seq = {choose_syev_band_nb(n, kd)};
    p.max_sweeps = -1;
    p.kd_work = 0;                              // 2*kd + nb_max
    return p;
}

} // namespace

template <Backend B, typename T>
Event syev_band(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& a_in,
                Span<typename base_type<T>::type> eigenvalues,
                JobType jobz,
                Uplo uplo,
                const Span<std::byte>& ws,
                StedcParams<typename base_type<T>::type> stedc_params,
                SyevBandParams params) {
    BATCHLAS_KERNEL_TRACE_SCOPE("syev_band.entry");
    validate_syev_band_dims(a_in, eigenvalues, jobz, uplo);

    if (!ctx.in_order()) {
        throw std::runtime_error("syev_band: requires an in-order Queue");
    }

    using Real = typename base_type<T>::type;

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    const int32_t kd = (params.kd > 0)
                           ? std::min(params.kd, std::max<int32_t>(1, n - 1))
                           : choose_syev_band_kd(n);
    const int32_t tau_sy2sb_n = std::max<int32_t>(0, n - kd);
    const SytrdBandReductionParams bandr =
        params.bandr_explicit ? params.bandr : make_bandr_params(n, kd);

    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    // ---- stage 1: dense -> band -------------------------------------------
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
    VectorView<T> tau_sy2sb_view(tau_sy2sb_span, tau_sy2sb_n, batch, 1, tau_sy2sb_n);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_band.sy2sb");
        const size_t sy2sb_ws_bytes =
            sytrd_sy2sb_buffer_size<B, T>(ctx, a, ab_view, tau_sy2sb_view, uplo, kd);
        auto sy2sb_ws = pool.allocate<std::byte>(ctx, sy2sb_ws_bytes);
        sytrd_sy2sb<B, T>(ctx, a, ab_view, tau_sy2sb_view, uplo, kd, sy2sb_ws);
    }

    // ---- stage 2: band -> tridiagonal (BANDR1) -----------------------------
    const int32_t em = std::max(0, n - 1);
    auto d_span = pool.allocate<Real>(ctx,
                                      static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    auto e_span = pool.allocate<Real>(ctx,
                                      static_cast<std::size_t>(em) * static_cast<std::size_t>(batch));
    auto tau_bandr_span = pool.allocate<T>(ctx,
                                           static_cast<std::size_t>(em) * static_cast<std::size_t>(batch));

    VectorView<Real> d_view(d_span, n, batch, 1, n);
    VectorView<Real> e_view(e_span, em, batch, 1, em);
    VectorView<T> tau_bandr_view(tau_bandr_span, em, batch, 1, em);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_band.bandr1");
        const size_t bandr_ws_bytes = sytrd_band_reduction_buffer_size<B, T>(
            ctx, ab_view, d_view, e_view, tau_bandr_view, uplo, kd, bandr);
        auto bandr_ws = pool.allocate<std::byte>(ctx, bandr_ws_bytes);
        sytrd_band_reduction<B, T>(ctx,
                                   ab_view,
                                   d_view,
                                   e_view,
                                   tau_bandr_view,
                                   uplo,
                                   kd,
                                   bandr_ws,
                                   bandr);
    }

    // ---- stage 3: tridiagonal eigensolve ----------------------------------
    VectorView<Real> evals_view(eigenvalues.data(), n, batch, 1, n);

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

    // STEDC is run in EigenVectors mode even though we only want eigenvalues.
    // Its divide-and-conquer merge relies on the eigenvector blocks, and its
    // NoEigenVectors path returns eigenvalues that drift by ~1e-2 (measured on
    // n=96..160, double) rather than the ~1e-14 the reductions deliver. The
    // same workaround is applied in syev_blocked; syev_two_stage still passes
    // NoEigenVectors here and is silently affected.
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_band.stedc");
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
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t syev_band_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& a,
                             JobType jobz,
                             Uplo uplo,
                             StedcParams<typename base_type<T>::type> stedc_params,
                             SyevBandParams params) {
    using Real = typename base_type<T>::type;

    if (jobz == JobType::EigenVectors) {
        throw std::invalid_argument(
            "syev_band_buffer_size: JobType::EigenVectors is not supported.");
    }

    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    const int32_t kd = (params.kd > 0)
                           ? std::min(params.kd, std::max<int32_t>(1, n - 1))
                           : choose_syev_band_kd(n);
    const int32_t tau_sy2sb_n = std::max<int32_t>(0, n - kd);
    const int32_t em = std::max(0, n - 1);
    const SytrdBandReductionParams bandr =
        params.bandr_explicit ? params.bandr : make_bandr_params(n, kd);

    // Mirror the allocation order in syev_band exactly: BumpAllocator sizing
    // must match the execution path or the pool overruns.
    size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<T>(ctx,
                                               static_cast<std::size_t>(kd + 1) *
                                                   static_cast<std::size_t>(n) *
                                                   static_cast<std::size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx,
                                               static_cast<std::size_t>(tau_sy2sb_n) *
                                                   static_cast<std::size_t>(batch));

    // Dummy views for the sub-buffer queries (sizes only depend on shapes).
    MatrixView<T, MatrixFormat::Dense> ab_dummy(nullptr,
                                                kd + 1,
                                                n,
                                                kd + 1,
                                                static_cast<int64_t>(kd + 1) * static_cast<int64_t>(n),
                                                batch);
    VectorView<T> tau_sy2sb_dummy(nullptr, tau_sy2sb_n, batch, 1, tau_sy2sb_n);
    VectorView<Real> d_dummy(nullptr, n, batch, 1, n);
    VectorView<Real> e_dummy(nullptr, em, batch, 1, em);
    VectorView<T> tau_bandr_dummy(nullptr, em, batch, 1, em);

    bytes += BumpAllocator::allocation_size<std::byte>(
        ctx, sytrd_sy2sb_buffer_size<B, T>(ctx, a, ab_dummy, tau_sy2sb_dummy, uplo, kd));

    bytes += BumpAllocator::allocation_size<Real>(
        ctx, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    bytes += BumpAllocator::allocation_size<Real>(
        ctx, static_cast<std::size_t>(em) * static_cast<std::size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(
        ctx, static_cast<std::size_t>(em) * static_cast<std::size_t>(batch));

    bytes += BumpAllocator::allocation_size<std::byte>(
        ctx,
        sytrd_band_reduction_buffer_size<B, T>(
            ctx, ab_dummy, d_dummy, e_dummy, tau_bandr_dummy, uplo, kd, bandr));

    bytes += BumpAllocator::allocation_size<Real>(ctx,
                                                  static_cast<std::size_t>(n) *
                                                      static_cast<std::size_t>(n) *
                                                      static_cast<std::size_t>(batch));
    bytes += BumpAllocator::allocation_size<std::byte>(
        ctx,
        stedc_workspace_size<B, Real>(ctx,
                                      static_cast<std::size_t>(n),
                                      static_cast<std::size_t>(batch),
                                      JobType::EigenVectors,
                                      stedc_params));

    return bytes;
}

#define SYEV_BAND_INSTANTIATE(back, fp) \
    template Event syev_band<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        JobType, \
        Uplo, \
        const Span<std::byte>&, \
        StedcParams<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        SyevBandParams); \
    template size_t syev_band_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        JobType, \
        Uplo, \
        StedcParams<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        SyevBandParams);

#define SYEV_BAND_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEV_BAND_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
SYEV_BAND_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYEV_BAND_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYEV_BAND_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SYEV_BAND_INSTANTIATE_FOR_BACKEND
#undef SYEV_BAND_INSTANTIATE

} // namespace batchlas

#include <batchlas/blas/extensions.hh>
#include "uplo_mirror.hh"
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/util/mempool.hh>

#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include "../queue.hh"
#include "../util/template-instantiations.hh"
#include "sytrd_sb2st_hh.hh"
#include "two_stage_common.hh"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace batchlas {

using namespace two_stage_detail;

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
    // Uplo::Upper is accepted; the solve mirrors it into Lower first. See uplo_mirror.hh.

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

    // Uplo::Upper: mirror into Lower, then run the ordinary Lower pipeline (sytrd_sy2sb ->
    // sb2st -> stedc -> back-transform), none of which implements Upper. See uplo_mirror.hh.
    if (uplo == Uplo::Upper) {
        mirror_upper_to_lower<B, T>(ctx, a_in);
        uplo = Uplo::Lower;
    }

    if constexpr (B == Backend::NETLIB) {
        if (jobz == JobType::EigenVectors) {
            return syev_blocked<B, T>(ctx, a_in, eigenvalues, jobz, uplo, ws, stedc_params);
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

    // Stage 2. BOTH modes use the Householder chase.
    //
    // This used to read "eigenvalues-only keeps the cheaper Givens chase". The
    // Givens chase is not cheaper -- it is ~5x more expensive on this GPU, and
    // the belief that it was is why eigenvalues-only, which does strictly less
    // work than eigenvector mode, measured 3.7-4x SLOWER than it at n=1024.
    //
    // The reason is occupancy, not arithmetic. Both chases are sequential per
    // matrix and parallel only over the batch, but sytrd_sb2st_hh was given a
    // 256-thread 2D lane mapping (sytrd_sb2st_hh.cc:103-106) while the Givens
    // path still runs one 32-lane sub-group per matrix
    // (sytrd_sb2st_cta.cc:391-394), with a mostly-serial `lid == 0` spine in the
    // kd > 32 fallback (sytrd_sb2st.cc:588-707). That is 8x fewer lanes per
    // matrix, and at batch 1 it is 32 threads on a 128-SM device.
    //
    // Measured, RTX 4090, float, n=1024, kd=32: Givens chase ~366 ms vs
    // Householder chase 67.5 ms, the latter essentially flat to batch 128.
    //
    // The cost of the switch is memory: eigenvalues-only now also allocates the
    // stage-2 reflectors V and their tau, which it discards. That is the same
    // workspace the eigenvector path has always allocated, and the buffer-size
    // query below is updated in lockstep.
    //
    // Only the *phase* chain stays eigenvector-only: it converts eigenvectors of
    // the tridiagonal built from |e| back to those of the signed one, and the
    // eigenvalues of the two are identical (a diagonal +-1 similarity).
    const bool use_givens = !want_eigvecs && two_stage_use_givens_chase_for_values();
    const auto sb2st_sched = use_givens ? std::vector<internal::Sb2stHhRefl>{}
                                        : internal::build_sb2st_hh_schedule(n, kd);
    const int32_t nrefl = static_cast<int32_t>(sb2st_sched.size());

    UnifiedVector<int32_t> sb2st_starts(static_cast<std::size_t>(nrefl));
    UnifiedVector<int32_t> sb2st_lens(static_cast<std::size_t>(nrefl));
    for (int32_t k = 0; k < nrefl; ++k) {
        sb2st_starts[k] = sb2st_sched[k].start;
        sb2st_lens[k] = sb2st_sched[k].len;
    }

    // Commuting runs of reflectors, so the back-transform can apply a whole run
    // at once. Lives until the back-transform's event completes.
    // Wave offsets are only consumed by the back-transform, which exists only in
    // eigenvector mode.
    const auto sb2st_wave_host = want_eigvecs
                                     ? internal::build_sb2st_hh_wave_offsets(sb2st_sched, n)
                                     : std::vector<int32_t>{};
    UnifiedVector<int32_t> sb2st_waves(sb2st_wave_host.size());
    for (std::size_t k = 0; k < sb2st_wave_host.size(); ++k) {
        sb2st_waves[k] = sb2st_wave_host[k];
    }

    Span<T> v_sb2st_span;
    Span<T> tau_sb2st_hh_span;
    Span<T> ab_tri_span;
    MatrixView<T, MatrixFormat::Dense> v_sb2st_view;
    VectorView<T> tau_sb2st_hh_view;
    MatrixView<T, MatrixFormat::Dense> ab_tri_view;

    if (use_givens) {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.sb2st");
        const size_t sb2st_ws_bytes = sytrd_sb2st_buffer_size<B, T>(
            ctx, ab_view, d_view, e_view, tau_sb2st_view, uplo, kd, sb2st_block_size);
        auto sb2st_ws = pool.allocate<std::byte>(ctx, sb2st_ws_bytes);
        sytrd_sb2st<B, T>(ctx, ab_view, d_view, e_view, tau_sb2st_view, uplo, kd,
                          sb2st_ws, sb2st_block_size);
    } else {
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

        // V and tau are dead in eigenvalues-only mode; only the back-transform
        // reads them. The phase chain is likewise eigenvector-only.
        if (want_eigvecs) {
            build_phase_from_kd1_band<T>(ctx, ab_tri_view, phase_view);
        }
    }
    (void)tau_sb2st_view;
    (void)sb2st_block_size;

    VectorView<Real> evals_view(eigenvalues.data(), n, batch, 1, n);

    if (!want_eigvecs) {
        BATCHLAS_KERNEL_TRACE_SCOPE("syev_two_stage.stebz_evals");
        auto m_span = pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
        StebzParams<Real> bp;
        bp.range = EigenRangeType::Index;
        bp.il = 0;
        bp.iu = n - 1;
        bp.order = SortOrder::Ascending;
        const size_t stebz_ws_bytes = stebz_buffer_size<B, Real>(ctx,
                                                                 static_cast<std::size_t>(n),
                                                                 static_cast<std::size_t>(batch),
                                                                 bp);
        auto stebz_ws = pool.allocate<std::byte>(ctx, stebz_ws_bytes);
        stebz<B, Real>(ctx, d_view, e_view, evals_view, m_span, stebz_ws, bp);

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
                                    Span<const int32_t>(sb2st_lens.data(), sb2st_lens.size()),
                                    Span<const int32_t>(sb2st_waves.data(), sb2st_waves.size()));
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
    // Uplo::Upper is accepted; workspace is identical, the mirror is in-place.

    if constexpr (B == Backend::NETLIB) {
        if (jobz == JobType::EigenVectors) {
            return syev_blocked_buffer_size<B, T>(ctx, a, jobz, uplo, stedc_params);
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
    // Stage-2 reflector storage is now allocated in BOTH modes: eigenvalues-only
    // also runs the Householder chase (see the note at its call site), and simply
    // discards V/tau. Only the phase chain and Z stay eigenvector-only.
    {
        const int32_t nr = std::max<int32_t>(1, internal::sb2st_hh_num_reflectors(n, kd));
        const int32_t kdw = internal::sb2st_hh_work_bandwidth(n, kd);
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
    }
    if (want_eigvecs) {
        bytes += BumpAllocator::allocation_size<T>(ctx,
                                                   static_cast<std::size_t>(n) *
                                                       static_cast<std::size_t>(batch)); // phase/sign chain
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

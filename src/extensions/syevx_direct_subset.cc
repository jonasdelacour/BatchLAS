// syevx_direct_subset: partial symmetric eigensolve via reduction to tridiagonal
// form, a subset tridiagonal solve, and a narrowed back-transform.
//
// This is SYEVX_PLAN.md Tier 2. Structure (compare syev_two_stage.cc):
//
//     sytrd_sy2sb   dense -> band
//     sytrd_sb2st   band  -> tridiagonal (d, e)
//     stebz         selected eigenvalues only          <- replaces full stedc
//     stein         selected eigenvectors only         <- replaces full stedc
//     ormqr_blocked back-transform, k columns not n    <- narrowed
//
// Where the saving comes from, relative to a full syev:
//
//   * the O(n^3) tridiagonal divide-and-conquer is replaced by an O(n*k) subset
//     solve, and
//   * the back-transform drops from 2n^3 to 2n^2*k.
//
// The reduction to tridiagonal form is O(n^3) either way and is not saved. That
// is what caps the achievable speedup at roughly 3x (SYEVX_PLAN.md §2.2).
//
// IMPORTANT (see two_stage_common.hh): with eigenvectors requested the reduction
// runs at kd = 1, so the band stage is a pure extract and there are no sb2st
// reflectors to undo. Only the stage-1 Q is back-transformed. Eigenvalue-only
// solves use a wide band and never need a back-transform at all.

#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <blas/linalg.hh>
#include <blas/functions.hh>
#include <internal/ormqr_blocked.hh>
#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>
#include "../util/template-instantiations.hh"
#include "two_stage_common.hh"

namespace batchlas {

template <Backend B, typename T, MatrixFormat MFormat>
struct SyevxSubsetFinalizeKernel;

namespace {

// Index range of the wanted block within the ascending spectrum.
struct WantedRange {
    int64_t il;
    int64_t iu;
};

inline WantedRange wanted_range(int64_t n, int64_t k, bool find_largest) {
    return find_largest ? WantedRange{n - k, n - 1} : WantedRange{0, k - 1};
}

} // namespace

template <Backend B, typename T, MatrixFormat MFormat>
Event syevx_direct_subset(Queue& ctx,
                          const MatrixView<T, MFormat>& A,
                          Span<typename base_type<T>::type> W,
                          size_t neigs,
                          Span<std::byte> workspace,
                          JobType jobz,
                          const MatrixView<T, MatrixFormat::Dense>& V,
                          const SyevxParams<T>& params) {
    using Real = typename base_type<T>::type;

    if constexpr (!syevx_direct_subset_supported<T, MFormat>()) {
        (void)ctx; (void)A; (void)W; (void)neigs; (void)workspace;
        (void)jobz; (void)V; (void)params;
        throw std::runtime_error(
            "syevx_direct_subset: only real scalar types with dense input are supported");
    } else {
        const int32_t n = static_cast<int32_t>(A.rows());
        const int32_t batch = static_cast<int32_t>(A.batch_size());
        const int64_t k = static_cast<int64_t>(neigs);
        const bool want_eigenvectors = (jobz == JobType::EigenVectors);

        if (A.rows() != A.cols()) throw std::runtime_error("syevx_direct_subset: A must be square");
        if (k < 1 || k > n) throw std::runtime_error("syevx_direct_subset: invalid neigs");
        if (!ctx.in_order()) {
            throw std::runtime_error("syevx_direct_subset: requires an in-order Queue");
        }

        using namespace two_stage_detail;
        const int32_t kd = choose_two_stage_kd_for_job(n, jobz);
        const int32_t tau_sy2sb_n = std::max<int32_t>(0, n - kd);
        const int32_t sb2st_block_size = choose_two_stage_sb2st_block_size();
        const int32_t p = std::max<int32_t>(0, n - 1);
        const int32_t ormqr_block_size = tuning::ormqr_block_size_for_n(n);

        BumpAllocator pool(workspace);

        // syevx must not modify A, and sy2sb overwrites its input.
        auto a_copy_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * n * batch);
        auto a_copy_ptrs = pool.allocate<T*>(ctx, static_cast<size_t>(batch));
        MatrixView<T, MatrixFormat::Dense> a(a_copy_span.data(), n, n, n,
                                             static_cast<int64_t>(n) * n, batch,
                                             a_copy_ptrs.data());
        MatrixView<T, MatrixFormat::Dense>::copy(ctx, a, A);

        // Stage 1: dense -> band.
        auto ab_span = pool.allocate<T>(ctx, static_cast<size_t>(kd + 1) * n * batch);
        MatrixView<T, MatrixFormat::Dense> ab_view(ab_span.data(), kd + 1, n, kd + 1,
                                                   static_cast<int64_t>(kd + 1) * n, batch);
        auto tau_sy2sb_span = pool.allocate<T>(ctx, static_cast<size_t>(tau_sy2sb_n) * batch);
        VectorView<T> tau_sy2sb_view(tau_sy2sb_span, tau_sy2sb_n, batch, 1, tau_sy2sb_n);

        Span<T> phase_span;
        VectorView<T> phase_view;
        if (want_eigenvectors) {
            phase_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * batch);
            phase_view = VectorView<T>(phase_span, n, batch, 1, n);
        }

        {
            const size_t bytes = sytrd_sy2sb_buffer_size<B, T>(ctx, a, ab_view, tau_sy2sb_view, Uplo::Lower, kd);
            auto sy2sb_ws = pool.allocate<std::byte>(ctx, bytes);
            sytrd_sy2sb<B, T>(ctx, a, ab_view, tau_sy2sb_view, Uplo::Lower, kd, sy2sb_ws);
        }

        if (want_eigenvectors) {
            build_phase_from_kd1_band<T>(ctx, ab_view, phase_view);
        }

        // Stage 2: band -> tridiagonal.
        auto d_span = pool.allocate<Real>(ctx, static_cast<size_t>(n) * batch);
        auto e_span = pool.allocate<Real>(ctx, static_cast<size_t>(std::max(0, n - 1)) * batch);
        auto tau_sb2st_span = pool.allocate<T>(ctx, static_cast<size_t>(std::max(0, n - 1)) * batch);
        VectorView<Real> d_view(d_span, n, batch, 1, n);
        VectorView<Real> e_view(e_span, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        VectorView<T> tau_sb2st_view(tau_sb2st_span, std::max(0, n - 1), batch, 1, std::max(0, n - 1));

        {
            const size_t bytes = sytrd_sb2st_buffer_size<B, T>(ctx, ab_view, d_view, e_view,
                                                               tau_sb2st_view, Uplo::Lower, kd,
                                                               sb2st_block_size);
            auto sb2st_ws = pool.allocate<std::byte>(ctx, bytes);
            sytrd_sb2st<B, T>(ctx, ab_view, d_view, e_view, tau_sb2st_view, Uplo::Lower, kd,
                              sb2st_ws, sb2st_block_size);
        }

        // Subset tridiagonal solve. Internally always ascending: stein's cluster
        // detection walks consecutive eigenvalues and requires that order.
        const auto range = wanted_range(n, k, params.find_largest);
        auto w_sub_span = pool.allocate<Real>(ctx, static_cast<size_t>(k) * batch);
        auto m_span = pool.allocate<int32_t>(ctx, static_cast<size_t>(batch));
        VectorView<Real> w_sub(w_sub_span, static_cast<int>(k), batch, 1, static_cast<int>(k));

        StebzParams<Real> bp;
        bp.range = EigenRangeType::Index;
        bp.il = range.il;
        bp.iu = range.iu;
        bp.order = SortOrder::Ascending;
        {
            const size_t bytes = stebz_buffer_size<B, Real>(ctx, n, batch, bp);
            auto stebz_ws = pool.allocate<std::byte>(ctx, bytes);
            stebz<B, Real>(ctx, d_view, e_view, w_sub, m_span, stebz_ws, bp);
        }

        if (want_eigenvectors) {
            // stein writes straight into V, so the only extra pass is the
            // ordering fix-up at the end.
            SteinParams<Real> sp;
            {
                const size_t bytes = stein_buffer_size<B, Real>(ctx, n, static_cast<size_t>(k), batch, sp);
                auto stein_ws = pool.allocate<std::byte>(ctx, bytes);
                auto V_sub = V({0, n}, {0, static_cast<int64_t>(k)});
                stein<B, Real>(ctx, d_view, e_view, w_sub, static_cast<size_t>(k), V_sub, stein_ws, sp);

                apply_phase_rows<Real>(ctx, V_sub,
                                       VectorView<Real>(phase_span.data(), n, batch, 1, n));

                // Back-transform through the stage-1 reflectors, applied to k
                // columns rather than n. This is the term Tier 2 exists to shrink.
                if (p > 0) {
                    auto aq_span = pool.allocate<T>(ctx, static_cast<size_t>(p) * p * batch);
                    auto tau_q_span = pool.allocate<T>(ctx, static_cast<size_t>(p) * batch);
                    MatrixView<T, MatrixFormat::Dense> aq_view(aq_span.data(), p, p, p,
                                                               static_cast<int64_t>(p) * p, batch);
                    VectorView<T> tau_q_view(tau_q_span, p, batch, 1, p);
                    pack_sytrd_lower_to_qsub_qr_layout<T>(ctx, a, aq_view, tau_sy2sb_view, tau_q_view, n);
                    Span<T> tau_q_flat(tau_q_span.data(), static_cast<size_t>(p) * batch);

                    auto v_sub_rows = V_sub({1, SliceEnd()}, Slice{});
                    size_t bytes_ormqr = 0;
                    if constexpr (B == Backend::NETLIB) {
                        bytes_ormqr = backend::ormqr_vendor_buffer_size<B, T>(
                            ctx, aq_view, v_sub_rows, Side::Left, Transpose::NoTrans, tau_q_flat);
                    } else {
                        bytes_ormqr = ormqr_blocked_buffer_size<B, T>(
                            ctx, aq_view, v_sub_rows, Side::Left, Transpose::NoTrans, tau_q_flat,
                            ormqr_block_size);
                    }
                    auto ormqr_ws = pool.allocate<std::byte>(ctx, bytes_ormqr);
                    if constexpr (B == Backend::NETLIB) {
                        backend::ormqr_vendor<B, T>(ctx, aq_view, v_sub_rows, Side::Left,
                                                    Transpose::NoTrans, tau_q_flat, ormqr_ws);
                    } else {
                        ormqr_blocked<B, T>(ctx, aq_view, v_sub_rows, Side::Left,
                                            Transpose::NoTrans, tau_q_flat, ormqr_ws,
                                            ormqr_block_size);
                    }
                }
            }
        }

        // Write the eigenvalues out in the requested order, reversing V's columns
        // in place when the largest were asked for.
        const bool reverse = params.find_largest;
        auto* w_out = W.data();
        const auto* w_in = w_sub_span.data();
        T* v_ptr = want_eigenvectors ? V.data_ptr() : nullptr;
        const int64_t v_ld = want_eigenvectors ? V.ld() : 0;
        const int64_t v_stride = want_eigenvectors ? V.stride() : 0;

        const size_t wg = std::min<size_t>(256, static_cast<size_t>(std::max<int64_t>(n, 1)));
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for<SyevxSubsetFinalizeKernel<B, T, MFormat>>(
                sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch) * wg}, sycl::range{wg}),
                [=](sycl::nd_item<1> item) {
                    const int64_t tid = static_cast<int64_t>(item.get_local_linear_id());
                    const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                    const int64_t local_size = static_cast<int64_t>(item.get_local_range(0));

                    for (int64_t i = tid; i < k; i += local_size) {
                        const int64_t src = reverse ? (k - 1 - i) : i;
                        w_out[bid * k + i] = w_in[bid * k + src];
                    }

                    if (v_ptr != nullptr && reverse) {
                        // Swap column pairs; the loop covers each pair once.
                        auto* vb = v_ptr + bid * v_stride;
                        const int64_t half = k / 2;
                        for (int64_t linear = tid; linear < n * half; linear += local_size) {
                            const int64_t row = linear % n;
                            const int64_t c = linear / n;
                            const int64_t c2 = k - 1 - c;
                            const T tmp = vb[row + c * v_ld];
                            vb[row + c * v_ld] = vb[row + c2 * v_ld];
                            vb[row + c2 * v_ld] = tmp;
                        }
                    }
                });
        });

        return ctx.get_event();
    }
}

template <Backend B, typename T, MatrixFormat MFormat>
size_t syevx_direct_subset_buffer_size(Queue& ctx,
                                       const MatrixView<T, MFormat>& A,
                                       Span<typename base_type<T>::type> W,
                                       size_t neigs,
                                       JobType jobz,
                                       const MatrixView<T, MatrixFormat::Dense>& V,
                                       const SyevxParams<T>& params) {
    using Real = typename base_type<T>::type;

    if constexpr (!syevx_direct_subset_supported<T, MFormat>()) {
        (void)ctx; (void)A; (void)W; (void)neigs; (void)jobz; (void)V; (void)params;
        return 0;
    } else {
        (void)W; (void)V; (void)params;
        const int32_t n = static_cast<int32_t>(A.rows());
        const int32_t batch = static_cast<int32_t>(A.batch_size());
        const int64_t k = static_cast<int64_t>(neigs);
        const bool want_eigenvectors = (jobz == JobType::EigenVectors);

        using namespace two_stage_detail;
        const int32_t kd = choose_two_stage_kd_for_job(n, jobz);
        const int32_t tau_sy2sb_n = std::max<int32_t>(0, n - kd);
        const int32_t sb2st_block_size = choose_two_stage_sb2st_block_size();
        const int32_t p = std::max<int32_t>(0, n - 1);
        const int32_t ormqr_block_size = tuning::ormqr_block_size_for_n(n);

        size_t bytes = 0;
        bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(n) * n * batch);
        bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
        bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(kd + 1) * n * batch);
        bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(tau_sy2sb_n) * batch);
        if (want_eigenvectors) {
            bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(n) * batch);
        }

        // Shapes must match the runtime views exactly so the provider choices agree.
        MatrixView<T, MatrixFormat::Dense> a_dummy(nullptr, n, n, n, static_cast<int64_t>(n) * n, batch);
        MatrixView<T, MatrixFormat::Dense> ab_dummy(nullptr, kd + 1, n, kd + 1,
                                                    static_cast<int64_t>(kd + 1) * n, batch);
        VectorView<T> tau_sy2sb_dummy(nullptr, tau_sy2sb_n, batch, 1, tau_sy2sb_n);
        VectorView<Real> d_dummy(nullptr, n, batch, 1, n);
        VectorView<Real> e_dummy(nullptr, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        VectorView<T> tau_sb2st_dummy(nullptr, std::max(0, n - 1), batch, 1, std::max(0, n - 1));

        bytes += BumpAllocator::allocation_size<std::byte>(
            ctx, sytrd_sy2sb_buffer_size<B, T>(ctx, a_dummy, ab_dummy, tau_sy2sb_dummy, Uplo::Lower, kd));

        bytes += BumpAllocator::allocation_size<Real>(ctx, static_cast<size_t>(n) * batch);
        bytes += BumpAllocator::allocation_size<Real>(ctx, static_cast<size_t>(std::max(0, n - 1)) * batch);
        bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(std::max(0, n - 1)) * batch);
        bytes += BumpAllocator::allocation_size<std::byte>(
            ctx, sytrd_sb2st_buffer_size<B, T>(ctx, ab_dummy, d_dummy, e_dummy, tau_sb2st_dummy,
                                                Uplo::Lower, kd, sb2st_block_size));

        bytes += BumpAllocator::allocation_size<Real>(ctx, static_cast<size_t>(k) * batch);
        bytes += BumpAllocator::allocation_size<int32_t>(ctx, static_cast<size_t>(batch));

        StebzParams<Real> bp;
        bp.range = EigenRangeType::Index;
        bytes += BumpAllocator::allocation_size<std::byte>(ctx, stebz_buffer_size<B, Real>(ctx, n, batch, bp));

        if (want_eigenvectors) {
            SteinParams<Real> sp;
            bytes += BumpAllocator::allocation_size<std::byte>(
                ctx, stein_buffer_size<B, Real>(ctx, n, static_cast<size_t>(k), batch, sp));

            if (p > 0) {
                bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(p) * p * batch);
                bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(p) * batch);

                MatrixView<T, MatrixFormat::Dense> aq_dummy(nullptr, p, p, p,
                                                            static_cast<int64_t>(p) * p, batch);
                // The back-transform target is the k-column block minus its first row.
                MatrixView<T, MatrixFormat::Dense> c_dummy(nullptr, p, static_cast<int>(k), n,
                                                           static_cast<int64_t>(n) * k, batch);
                Span<T> tau_q_flat(nullptr, static_cast<size_t>(p) * batch);
                size_t bytes_ormqr = 0;
                if constexpr (B == Backend::NETLIB) {
                    bytes_ormqr = backend::ormqr_vendor_buffer_size<B, T>(
                        ctx, aq_dummy, c_dummy, Side::Left, Transpose::NoTrans, tau_q_flat);
                } else {
                    bytes_ormqr = ormqr_blocked_buffer_size<B, T>(
                        ctx, aq_dummy, c_dummy, Side::Left, Transpose::NoTrans, tau_q_flat,
                        ormqr_block_size);
                }
                bytes += BumpAllocator::allocation_size<std::byte>(ctx, bytes_ormqr);
            }
        }

        return bytes;
    }
}

#define SYEVX_SUBSET_INSTANTIATE(back, fp, fmt) \
    template Event syevx_direct_subset<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);\
    template size_t syevx_direct_subset_buffer_size<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);

#define SYEVX_SUBSET_INSTANTIATE_FOR_BACKEND_TYPE(back, fp) \
    BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(SYEVX_SUBSET_INSTANTIATE, back, fp)

#define SYEVX_SUBSET_INSTANTIATE_FOR_BACKEND(back)\
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEVX_SUBSET_INSTANTIATE_FOR_BACKEND_TYPE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    SYEVX_SUBSET_INSTANTIATE_FOR_BACKEND(Backend::CUDA);
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    SYEVX_SUBSET_INSTANTIATE_FOR_BACKEND(Backend::ROCM);
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    SYEVX_SUBSET_INSTANTIATE_FOR_BACKEND(Backend::NETLIB);
#endif

#undef SYEVX_SUBSET_INSTANTIATE_FOR_BACKEND
#undef SYEVX_SUBSET_INSTANTIATE_FOR_BACKEND_TYPE
#undef SYEVX_SUBSET_INSTANTIATE

} // namespace batchlas

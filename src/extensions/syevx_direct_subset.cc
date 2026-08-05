// syevx_direct_subset: partial symmetric eigensolve via reduction to tridiagonal
// form, a subset tridiagonal solve, and a narrowed back-transform.
//
// This is SYEVX_PLAN.md Tier 2. Structure (compare syev_two_stage.cc):
//
//     sytrd_sy2sb    dense -> band
//     sytrd_sb2st_hh band  -> tridiagonal (d, e), retaining Q2
//     stebz          selected eigenvalues only          <- replaces full stedc
//     stein          selected eigenvectors only         <- replaces full stedc
//     unmqr_hb2st    Q2 back-transform, k columns not n <- narrowed
//     ormqr_blocked  Q1 back-transform, k columns not n <- narrowed
//
// Where the saving comes from, relative to a full syev:
//
//   * the O(n^3) tridiagonal divide-and-conquer is replaced by an O(n*k) subset
//     solve, and
//   * both back-transforms drop from O(n^3) to O(n^2*k).
//
// The reduction to tridiagonal form is O(n^3) either way and is not saved. That
// is what caps the achievable speedup at roughly 3x (SYEVX_PLAN.md §2.2).
//
// On kd: this path used to pin kd = 1 when eigenvectors were wanted, because the
// Givens sytrd_sb2st discards Q2 and a kd = 1 band makes stage 2 a pure extract.
// That made stage 1 an unblocked BLAS-2 reduction -- the dominant cost here, done
// the slow way. sytrd_sb2st_hh retains Q2, so the clamp is gone and both modes now
// reduce at a real band width.
//
// On the stage-2 chase: BOTH modes now use the Householder chase. This file used
// to say eigenvalue-only solves "keep the cheaper Givens chase". The Givens chase
// is not cheaper -- profiled at n = 1024 on an RTX 4090 it is 348.4 ms/call
// against 62.4 ms for sytrd_sb2st_hh. The cause is occupancy, not arithmetic:
// both chases are sequential per matrix and parallel only over the batch, but
// sytrd_sb2st_hh got a 256-thread 2D lane mapping while the Givens path still
// runs one 32-lane sub-group per matrix. This is the same fix already applied to
// syev_two_stage.cc; see the long note at its call site and
// two_stage_common.hh::two_stage_use_givens_chase_for_values.
//
// Eigenvalues-only therefore also allocates the stage-2 reflectors V and their
// tau, and discards them. That is the same workspace the eigenvector path has
// always allocated, and syevx_direct_subset_buffer_size below is updated in
// lockstep. Only the *phase* chain stays eigenvector-only: it converts
// eigenvectors of the tridiagonal built from |e| back to those of the signed one,
// and the eigenvalues of the two are identical (a diagonal +-1 similarity), so
// stebz does not need it.
//
// BATCHLAS_SYEV_TWO_STAGE_CHASE=givens restores the old eigenvalues-only path,
// making this an intra-run A/B rather than a comparison across builds.

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
#include "sytrd_sb2st_hh.hh"
#include <vector>

namespace batchlas {

template <Backend B, typename T, MatrixFormat MFormat>
struct SyevxSubsetFinalizeKernel;

namespace {

// Per-item length of the internal stebz eigenvalue buffer.
//
// For an index block stebz produces exactly `rr.max_count` values, which is what
// the caller's capacity already covers. For a value range the count is
// data-dependent and bounded only by n, and stebz throws outright unless its `w`
// holds n entries per item (src/extensions/stebz.cc:94-98) -- so the internal
// buffer is widened to max(n, capacity) independently of the caller's capacity.
// The max() with capacity keeps `stein`'s `w.size() >= k` precondition satisfied
// when a caller declares a capacity above n.
//
// THIS EXPRESSION IS DUPLICATED in syevx_direct_subset_buffer_size below and the
// two must stay identical -- it is a BumpAllocator size, and this repo's memory
// records that an under-computed workspace size is the failure mode here.
inline size_t subset_w_sub_len(int64_t n, int64_t capacity, bool value_range) {
    return value_range ? static_cast<size_t>(std::max<int64_t>(n, capacity))
                       : static_cast<size_t>(capacity);
}

} // namespace

template <Backend B, typename T, MatrixFormat MFormat>
Event syevx_direct_subset(Queue& ctx,
                          const MatrixView<T, MFormat>& A,
                          Span<typename base_type<T>::type> W,
                          Span<int32_t> m,
                          size_t neigs,
                          Span<std::byte> workspace,
                          JobType jobz,
                          const MatrixView<T, MatrixFormat::Dense>& V,
                          const SyevxParams<T>& params) {
    using Real = typename base_type<T>::type;

    if constexpr (!syevx_direct_subset_supported<T, MFormat>()) {
        (void)ctx; (void)A; (void)W; (void)m; (void)neigs; (void)workspace;
        (void)jobz; (void)V; (void)params;
        throw std::runtime_error(
            "syevx_direct_subset: only real scalar types with dense input are supported");
    } else {
        const int32_t n = static_cast<int32_t>(A.rows());
        const int32_t batch = static_cast<int32_t>(A.batch_size());
        // `capacity` is the caller's declared room per item: the stride of W and
        // the column count of V. It is deliberately NOT the number of eigenvalues
        // produced (that is the per-item `count` the finalize kernel reads out of
        // m_span) nor the stride of the internal stebz buffer (`w_sub_len`).
        // Conflating those three -- they used to be one `k` -- is how a batched
        // solver writes item b's answers into item b+1's slots.
        const int64_t capacity = static_cast<int64_t>(neigs);
        const bool want_eigenvectors = (jobz == JobType::EigenVectors);

        if (A.rows() != A.cols()) throw std::runtime_error("syevx_direct_subset: A must be square");
        // A capacity above n is no longer rejected: `neigs` is a capacity now, so
        // an over-large one just leaves the tail of W and V unwritten. The work
        // count is clamped to n inside syevx_resolve_range instead. Zero capacity
        // is still rejected -- stein requires k >= 1 and the finalize geometry is
        // untested at zero.
        if (capacity < 1) throw std::runtime_error("syevx_direct_subset: invalid neigs");
        if (!m.empty() && static_cast<int64_t>(m.size()) < batch) {
            throw std::runtime_error("syevx_direct_subset: m must cover every batch item");
        }
        if (!ctx.in_order()) {
            throw std::runtime_error("syevx_direct_subset: requires an in-order Queue");
        }

        // Resolved once, here, and used for both the stebz range and the workspace
        // sizes so that this function and syevx_direct_subset_buffer_size can never
        // disagree about what was asked for.
        const auto rr = syevx_resolve_range(n, neigs, params);
        const size_t w_sub_len = subset_w_sub_len(n, capacity, rr.value_range);

        using namespace two_stage_detail;
        const int32_t kd = choose_two_stage_kd_for_job(n, jobz);
        const int32_t tau_sy2sb_n = std::max<int32_t>(0, n - kd);
        const int32_t sb2st_block_size = choose_two_stage_sb2st_block_size();
        const int32_t ormqr_block_size = tuning::ormqr_block_size_for_n(n);

        // Which stage-2 chase? Householder unless the A/B env var asks for the
        // old Givens path, which is only legal without eigenvectors (it discards
        // Q2). See the note at the top of this file.
        const bool use_givens =
            !want_eigenvectors && two_stage_use_givens_chase_for_values();

        // Stage-2 reflector schedule. Depends only on (n, kd) -- never on the
        // matrix values -- so it is identical for every batch item and can be
        // replayed on the host.
        const auto sb2st_sched = use_givens
                                     ? std::vector<internal::Sb2stHhRefl>{}
                                     : internal::build_sb2st_hh_schedule(n, kd);
        const int32_t nrefl = static_cast<int32_t>(sb2st_sched.size());
        // starts/lens/waves are consumed only by the Q2 back-transform, which
        // exists only in eigenvector mode. Eigenvalues-only runs the same chase
        // but never reads the schedule on the device.
        const int32_t nrefl_dev = want_eigenvectors ? nrefl : 0;
        UnifiedVector<int32_t> sb2st_starts(static_cast<size_t>(nrefl_dev));
        UnifiedVector<int32_t> sb2st_lens(static_cast<size_t>(nrefl_dev));
        for (int32_t i = 0; i < nrefl_dev; ++i) {
            sb2st_starts[i] = sb2st_sched[i].start;
            sb2st_lens[i] = sb2st_sched[i].len;
        }
        const auto sb2st_wave_host = want_eigenvectors
                                         ? internal::build_sb2st_hh_wave_offsets(sb2st_sched, n)
                                         : std::vector<int32_t>{};
        UnifiedVector<int32_t> sb2st_waves(sb2st_wave_host.size());
        for (size_t i = 0; i < sb2st_wave_host.size(); ++i) {
            sb2st_waves[i] = sb2st_wave_host[i];
        }

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

        // Stage 2: band -> tridiagonal. Both modes use the Householder chase --
        // eigenvector mode needs Q2 to apply to the selected vectors, and
        // eigenvalues-only uses it because it is ~5x faster than the Givens chase
        // even though it then throws Q2 away. See the top of this file.
        auto d_span = pool.allocate<Real>(ctx, static_cast<size_t>(n) * batch);
        auto e_span = pool.allocate<Real>(ctx, static_cast<size_t>(std::max(0, n - 1)) * batch);
        VectorView<Real> d_view(d_span, n, batch, 1, n);
        VectorView<Real> e_view(e_span, std::max(0, n - 1), batch, 1, std::max(0, n - 1));

        Span<T> v_sb2st_span;
        Span<T> tau_sb2st_hh_span;
        MatrixView<T, MatrixFormat::Dense> v_sb2st_view;
        VectorView<T> tau_sb2st_hh_view;

        if (!use_givens) {
            const int32_t nr = std::max<int32_t>(1, nrefl);
            v_sb2st_span = pool.allocate<T>(ctx, static_cast<size_t>(kd) * nr * batch);
            tau_sb2st_hh_span = pool.allocate<T>(ctx, static_cast<size_t>(nr) * batch);
            auto ab_tri_span = pool.allocate<T>(ctx, static_cast<size_t>(2) * n * batch);
            v_sb2st_view = MatrixView<T, MatrixFormat::Dense>(
                v_sb2st_span.data(), kd, nr, kd, static_cast<int64_t>(kd) * nr, batch);
            tau_sb2st_hh_view = VectorView<T>(tau_sb2st_hh_span, nr, batch, 1, nr);
            MatrixView<T, MatrixFormat::Dense> ab_tri_view(
                ab_tri_span.data(), 2, n, 2, static_cast<int64_t>(2) * n, batch);

            const size_t bytes = internal::sytrd_sb2st_hh_buffer_size<B, T>(ctx, n, kd, batch);
            auto hh_ws = pool.allocate<std::byte>(ctx, bytes);
            internal::sytrd_sb2st_hh<B, T>(ctx, ab_view, ab_tri_view, d_view, e_view,
                                           v_sb2st_view, tau_sb2st_hh_view, Uplo::Lower, kd,
                                           hh_ws);

            // The phase comes from stage 2's *output* tridiagonal, not from the
            // stage-1 band: it converts eigenvectors of the real tridiagonal
            // built from |e| back to those of the signed one. stebz works on |e|
            // directly -- the two tridiagonals are a diagonal +-1 similarity and
            // have identical eigenvalues -- so eigenvalues-only skips it, along
            // with V/tau, which only the back-transform reads.
            if (want_eigenvectors) {
                build_phase_from_kd1_band<T>(ctx, ab_tri_view, phase_view);
            }
        } else {
            auto tau_sb2st_span = pool.allocate<T>(ctx, static_cast<size_t>(std::max(0, n - 1)) * batch);
            VectorView<T> tau_sb2st_view(tau_sb2st_span, std::max(0, n - 1), batch, 1,
                                         std::max(0, n - 1));
            const size_t bytes = sytrd_sb2st_buffer_size<B, T>(ctx, ab_view, d_view, e_view,
                                                               tau_sb2st_view, Uplo::Lower, kd,
                                                               sb2st_block_size);
            auto sb2st_ws = pool.allocate<std::byte>(ctx, bytes);
            sytrd_sb2st<B, T>(ctx, ab_view, d_view, e_view, tau_sb2st_view, Uplo::Lower, kd,
                              sb2st_ws, sb2st_block_size);
        }

        // Subset tridiagonal solve. Internally always ascending: stein's cluster
        // detection walks consecutive eigenvalues and requires that order, so
        // bp.order is pinned to Ascending for EVERY range and params.order is
        // honoured only by the finalize kernel at the bottom of this function.
        // Asking stebz for Descending mid-chain would silently corrupt stein's
        // cluster grouping.
        auto w_sub_span = pool.allocate<Real>(ctx, w_sub_len * batch);
        auto m_span = pool.allocate<int32_t>(ctx, static_cast<size_t>(batch));
        VectorView<Real> w_sub(w_sub_span, static_cast<int>(w_sub_len), batch, 1,
                               static_cast<int>(w_sub_len));

        StebzParams<Real> bp;
        bp.range = rr.value_range ? EigenRangeType::Value : EigenRangeType::Index;
        bp.il = rr.il;
        bp.iu = rr.iu;
        bp.vl = params.vl;
        bp.vu = params.vu;
        bp.abstol = params.abstol;
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
                const size_t bytes = stein_buffer_size<B, Real>(ctx, n, static_cast<size_t>(capacity), batch, sp);
                auto stein_ws = pool.allocate<std::byte>(ctx, bytes);
                auto V_sub = V({0, n}, {0, capacity});
                // Per-item counts, straight from the m stebz just wrote -- read on
                // the device, so no host sync is introduced between the two calls.
                // Span has no Span<T> -> Span<const T> converting constructor, so
                // this must be spelled explicitly (same as the reflector schedule
                // spans below).
                //
                // For an index range every count equals the block size and this is
                // exactly the old uniform-k behaviour; for a value range it stops
                // inverse iteration at m[b] and, critically, makes stein ZERO the
                // columns [m[b], capacity) rather than leave stale workspace there.
                // The back-transforms below rely on that.
                stein<B, Real>(ctx, d_view, e_view, w_sub, static_cast<size_t>(capacity),
                               Span<const int32_t>(m_span.data(), m_span.size()),
                               V_sub, stein_ws, sp);

                apply_phase_rows<Real>(ctx, V_sub,
                                       VectorView<Real>(phase_span.data(), n, batch, 1, n));

                // V := Q2 V, over `capacity` columns rather than n.
                //
                // Both back-transforms stay at the UNIFORM column count even for a
                // value range where item b found only m[b] < capacity eigenvalues.
                // Columns [m[b], capacity) hold the zeros stein wrote and an
                // orthogonal transform maps zero to zero, so nothing propagates.
                //
                // The tradeoff, stated because it is real: if one item finds 3 and
                // another finds 200, every item pays the 200-column transform. The
                // alternative -- reading max_b m[b] back to the host to shape the
                // call -- would add a device->host sync per call, which is exactly
                // the defect SYEVX_PLAN.md §7.1 catalogues in LOBPCG. Uniform shape
                // across the batch is also what keeps these two kernels fast. A
                // caller who cares should use an index range or split the batch.
                if (nrefl > 0) {
                    internal::unmqr_hb2st<B, T>(
                        ctx, v_sb2st_view, tau_sb2st_hh_view, V_sub, n, kd,
                        Span<const int32_t>(sb2st_starts.data(), sb2st_starts.size()),
                        Span<const int32_t>(sb2st_lens.data(), sb2st_lens.size()),
                        Span<const int32_t>(sb2st_waves.data(), sb2st_waves.size()));
                }

                // V(kd:, :) := Q1 V(kd:, :), again over `capacity` columns rather
                // than n. Narrowing these two back-transforms from n to capacity
                // columns is the term Tier 2 exists to shrink.
                //
                // sy2sb factors panel i with GEQRF starting at row i+kd, so
                // a(kd:, 0:n-kd) is already a GEQRF-style reflector layout and the
                // sliced view suffices -- no packed copy, matching syev_two_stage.
                if (tau_sy2sb_n > 0) {
                    auto v1_view = a({kd, SliceEnd()}, {0, tau_sy2sb_n});
                    auto v_sub_rows = V_sub({kd, SliceEnd()}, Slice{});
                    Span<T> tau1_flat(tau_sy2sb_span.data(),
                                      static_cast<size_t>(tau_sy2sb_n) * batch);

                    size_t bytes_ormqr = 0;
                    if constexpr (B == Backend::NETLIB) {
                        bytes_ormqr = backend::ormqr_vendor_buffer_size<B, T>(
                            ctx, v1_view, v_sub_rows, Side::Left, Transpose::NoTrans, tau1_flat);
                    } else {
                        bytes_ormqr = ormqr_blocked_buffer_size<B, T>(
                            ctx, v1_view, v_sub_rows, Side::Left, Transpose::NoTrans, tau1_flat,
                            ormqr_block_size);
                    }
                    auto ormqr_ws = pool.allocate<std::byte>(ctx, bytes_ormqr);
                    if constexpr (B == Backend::NETLIB) {
                        backend::ormqr_vendor<B, T>(ctx, v1_view, v_sub_rows, Side::Left,
                                                    Transpose::NoTrans, tau1_flat, ormqr_ws);
                    } else {
                        ormqr_blocked<B, T>(ctx, v1_view, v_sub_rows, Side::Left,
                                            Transpose::NoTrans, tau1_flat, ormqr_ws,
                                            ormqr_block_size);
                    }
                }
            }
        }

        // Write the eigenvalues out in the requested order, reversing V's columns
        // in place when a descending block was asked for.
        //
        // `reverse` comes from the RESOLVED range, not from params.find_largest:
        // find_largest is meaningful only for SyevxSelect::Extremal, and an Index
        // or Value range takes its order from params.order. Deriving it from
        // find_largest here -- as this kernel used to -- is a second, independent
        // place the extremal assumption was baked in, and it would have quietly
        // returned an interior block in the wrong order.
        const bool reverse = rr.reverse;
        auto* w_out = W.data();
        const auto* w_in = w_sub_span.data();
        T* v_ptr = want_eigenvectors ? V.data_ptr() : nullptr;
        const int64_t v_ld = want_eigenvectors ? V.ld() : 0;
        const int64_t v_stride = want_eigenvectors ? V.stride() : 0;
        // Two strides that used to be one `k` and must not be re-merged: the output
        // side keeps the caller's capacity, the input side uses the internal
        // (possibly wider) stebz buffer.
        const int64_t w_in_stride = static_cast<int64_t>(w_sub_len);
        const int32_t* m_in = m_span.data();
        int32_t* m_out = m.empty() ? nullptr : m.data();

        const size_t wg = std::min<size_t>(256, static_cast<size_t>(std::max<int64_t>(n, 1)));
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for<SyevxSubsetFinalizeKernel<B, T, MFormat>>(
                sycl::nd_range<1>(sycl::range{static_cast<size_t>(batch) * wg}, sycl::range{wg}),
                [=](sycl::nd_item<1> item) {
                    const int64_t tid = static_cast<int64_t>(item.get_local_linear_id());
                    const int64_t bid = static_cast<int64_t>(item.get_group_linear_id());
                    const int64_t local_size = static_cast<int64_t>(item.get_local_range(0));

                    // True count for this item, as stebz found it. For an index
                    // block this is the block size; for a value range it is data
                    // dependent and may exceed the capacity. Read with signed
                    // arithmetic and clamped at zero: stebz does not itself clamp a
                    // Sturm-count difference, so a swapped interval can leave a
                    // negative here and an unsigned loop bound would be an enormous
                    // write.
                    const int64_t raw_count = static_cast<int64_t>(m_in[bid]);
                    const int64_t count = (raw_count > 0) ? raw_count : int64_t(0);
                    // Only the VALID PREFIX is written and reversed; slots
                    // [write, capacity) of W are left untouched, and so are the
                    // matching columns of V (which stein already zeroed).
                    const int64_t write = (count < capacity) ? count : capacity;

                    for (int64_t i = tid; i < write; i += local_size) {
                        const int64_t src = reverse ? (write - 1 - i) : i;
                        w_out[bid * capacity + i] = w_in[bid * w_in_stride + src];
                    }

                    if (v_ptr != nullptr && reverse) {
                        // Swap column pairs; the loop covers each pair once.
                        auto* vb = v_ptr + bid * v_stride;
                        const int64_t half = write / 2;
                        for (int64_t linear = tid; linear < n * half; linear += local_size) {
                            const int64_t row = linear % n;
                            const int64_t c = linear / n;
                            const int64_t c2 = write - 1 - c;
                            const T tmp = vb[row + c * v_ld];
                            vb[row + c * v_ld] = vb[row + c2 * v_ld];
                            vb[row + c2 * v_ld] = tmp;
                        }
                    }

                    if (m_out != nullptr && tid == 0) {
                        // The TRUE count, not `write`: m[b] > capacity is the
                        // caller's truncation signal.
                        m_out[bid] = static_cast<int32_t>(count);
                    }
                });
        });

        // sb2st_starts/lens/waves are read by unmqr_hb2st's kernels, not copied,
        // and their UnifiedVector destructors sycl::free immediately. Returning
        // without waiting would free them out from under a kernel still in flight.
        if (nrefl_dev > 0) ctx.wait();

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
        (void)W; (void)V;
        const int32_t n = static_cast<int32_t>(A.rows());
        const int32_t batch = static_cast<int32_t>(A.batch_size());
        const int64_t capacity = static_cast<int64_t>(neigs);
        const bool want_eigenvectors = (jobz == JobType::EigenVectors);

        // Same resolution, same expression, as the solver above. A Value range
        // widens the internal stebz buffer to n entries per item independently of
        // the caller's capacity, so this function is NOT range-independent and
        // must not be simplified back to `neigs`.
        const auto rr = syevx_resolve_range(n, neigs, params);
        const size_t w_sub_len = subset_w_sub_len(n, capacity, rr.value_range);

        using namespace two_stage_detail;
        const int32_t kd = choose_two_stage_kd_for_job(n, jobz);
        const int32_t tau_sy2sb_n = std::max<int32_t>(0, n - kd);
        const int32_t sb2st_block_size = choose_two_stage_sb2st_block_size();
        const int32_t ormqr_block_size = tuning::ormqr_block_size_for_n(n);
        // Must mirror the runtime chase selection exactly: eigenvalues-only also
        // runs the Householder chase now (unless BATCHLAS_SYEV_TWO_STAGE_CHASE=
        // givens), so it allocates V/tau/ab_tri too.
        const bool use_givens =
            !want_eigenvectors && two_stage_use_givens_chase_for_values();
        const int32_t nrefl = use_givens ? 0 : internal::sb2st_hh_num_reflectors(n, kd);

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

        if (!use_givens) {
            // Householder chase: reflectors, taus and the 2 x n tridiagonal band.
            const int32_t nr = std::max<int32_t>(1, nrefl);
            bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(kd) * nr * batch);
            bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(nr) * batch);
            bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(2) * n * batch);
            bytes += BumpAllocator::allocation_size<std::byte>(
                ctx, internal::sytrd_sb2st_hh_buffer_size<B, T>(ctx, n, kd, batch));
        } else {
            bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(std::max(0, n - 1)) * batch);
            bytes += BumpAllocator::allocation_size<std::byte>(
                ctx, sytrd_sb2st_buffer_size<B, T>(ctx, ab_dummy, d_dummy, e_dummy, tau_sb2st_dummy,
                                                    Uplo::Lower, kd, sb2st_block_size));
        }

        // w_sub, then m. Mirrors the runtime allocation order exactly.
        bytes += BumpAllocator::allocation_size<Real>(ctx, w_sub_len * batch);
        bytes += BumpAllocator::allocation_size<int32_t>(ctx, static_cast<size_t>(batch));

        StebzParams<Real> bp;
        bp.range = rr.value_range ? EigenRangeType::Value : EigenRangeType::Index;
        bytes += BumpAllocator::allocation_size<std::byte>(ctx, stebz_buffer_size<B, Real>(ctx, n, batch, bp));

        if (want_eigenvectors) {
            SteinParams<Real> sp;
            // Sized on the CAPACITY, which is what the solver passes as stein's k:
            // stein's scratch grid is n * k * batch whether or not a column ends up
            // used, so per-item counts do not shrink it.
            bytes += BumpAllocator::allocation_size<std::byte>(
                ctx, stein_buffer_size<B, Real>(ctx, n, static_cast<size_t>(capacity), batch, sp));

            // Q1 back-transform. Shapes mirror the runtime slices exactly:
            // v1 = a(kd:, 0:n-kd) and the target is V(kd:, 0:capacity).
            if (tau_sy2sb_n > 0) {
                const int32_t rows_below_kd = std::max<int32_t>(0, n - kd);
                MatrixView<T, MatrixFormat::Dense> v1_dummy(nullptr, rows_below_kd, tau_sy2sb_n, n,
                                                            static_cast<int64_t>(n) * n, batch);
                MatrixView<T, MatrixFormat::Dense> c_dummy(nullptr, rows_below_kd,
                                                           static_cast<int>(capacity), n,
                                                           static_cast<int64_t>(n) * capacity, batch);
                Span<T> tau1_flat(nullptr, static_cast<size_t>(tau_sy2sb_n) * batch);
                size_t bytes_ormqr = 0;
                if constexpr (B == Backend::NETLIB) {
                    bytes_ormqr = backend::ormqr_vendor_buffer_size<B, T>(
                        ctx, v1_dummy, c_dummy, Side::Left, Transpose::NoTrans, tau1_flat);
                } else {
                    bytes_ormqr = ormqr_blocked_buffer_size<B, T>(
                        ctx, v1_dummy, c_dummy, Side::Left, Transpose::NoTrans, tau1_flat,
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
        Span<int32_t>,\
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

#pragma once

#include <stdexcept>

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>

#include <blas/linalg.hh>
#include <blas/extensions.hh>

#include <blas/dispatch/context.hh>
#include <blas/dispatch/env.hh>
#include <blas/dispatch/provider.hh>

namespace batchlas {

template <Backend B, typename T>
Event syev(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& descrA, // A is overwritten with eigenvectors
           Span<typename base_type<T>::type> eigenvalues,
           JobType jobtype,
           Uplo uplo,
           Span<std::byte> workspace);

template <Backend B, typename T>
size_t syev_buffer_size(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        Span<typename base_type<T>::type> eigenvalues,
                        JobType jobtype,
                        Uplo uplo);

template <Backend B, typename T>
inline Event syev(Queue& ctx,
                  const Matrix<T, MatrixFormat::Dense>& descrA,
                  Span<typename base_type<T>::type> eigenvalues,
                  JobType jobtype,
                  Uplo uplo,
                  Span<std::byte> workspace) {
    return syev<B, T>(ctx, MatrixView<T, MatrixFormat::Dense>(descrA), eigenvalues, jobtype, uplo, workspace);
}

template <Backend B, typename T>
inline size_t syev_buffer_size(Queue& ctx,
                               const Matrix<T, MatrixFormat::Dense>& A,
                               Span<typename base_type<T>::type> eigenvalues,
                               JobType jobtype,
                               Uplo uplo) {
    return syev_buffer_size<B, T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), eigenvalues, jobtype, uplo);
}

} // namespace batchlas

namespace batchlas::backend {

// Implemented by backend wrapper TUs (e.g. cuSOLVER / rocSOLVER / LAPACKE).
template <Backend B, typename T>
Event syev_vendor(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& descrA,
                  Span<typename base_type<T>::type> eigenvalues,
                  JobType jobtype,
                  Uplo uplo,
                  Span<std::byte> workspace);

template <Backend B, typename T>
size_t syev_vendor_buffer_size(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& descrA,
                               Span<typename base_type<T>::type> eigenvalues,
                               JobType jobtype,
                               Uplo uplo);

} // namespace batchlas::backend

namespace batchlas::blas::dispatch {

namespace detail {

template <typename T>
inline SteqrParams<T> syev_cta_steqr_params(JobType jobtype) {
    SteqrParams<T> params{};
    // CTA STEQR defaults are tuned for speed; for some ill-conditioned small
    // projected problems (e.g. Rayleigh–Ritz in iterative eigensolvers) we want
    // a bit more robustness.
    params.max_sweeps = 400;
    // Wilkinson shifts tend to converge faster on tough small problems.
    // This is especially important when SYEV is used inside an outer
    // iterative eigensolver (syevx), where slow/incorrect Ritz solves can
    // lead to stagnation.
    params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
    return params;
}

template <typename T>
inline bool syev_supports_cta(const DeviceCaps& caps, const MatrixView<T, MatrixFormat::Dense>& A) {
    const int64_t n = A.rows();
    if (A.rows() != A.cols()) return false;
    if (n < 1 || n > 32) return false;
    // CTA supports small sizes (n<=32). Note: some sizes may be slower than others,
    // but this predicate is about functional support, not performance heuristics.
    if (!caps.is_gpu) return false;
    if (caps.max_sub_group < 32) return false;
    return true;
}

template <typename T>
inline bool syev_supports_blocked(const DeviceCaps& caps,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  Uplo uplo) {
    if (A.rows() != A.cols()) return false;
    if (A.rows() < 1 || A.batch_size() < 1) return false;
    if (!caps.is_gpu) return false;
    if (uplo != Uplo::Lower) return false;
    return true;
}

template <typename T>
inline bool syev_supports_two_stage(const DeviceCaps& caps,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    Uplo uplo) {
    if (A.rows() != A.cols()) return false;
    if (A.rows() < 1 || A.batch_size() < 1) return false;
    if (!caps.is_gpu) return false;
    if (uplo != Uplo::Lower) return false;
    return true;
}

inline Provider normalize_vendor_like(Provider p) {
    if (p == Provider::Netlib) return Provider::Vendor;
    return p;
}

// Should Auto prefer the vendor eigensolver over the BatchLAS blocked one?
//
// The default order below lists BatchLAS_Blocked ahead of Vendor, which made
// `Auto` pick the blocked path for every GPU matrix with n > 32 -- it is the
// first entry whose support predicate is unconditionally true on a GPU. That was
// an ordering, not a performance decision, and it is wrong nearly everywhere.
//
// The blocked reduction's panel factorization (LatrdLowerPanelKernel) is
// parallel over the *batch*: one work-group per matrix. At small batch it
// therefore runs on a handful of SMs. Profiled at n=1024, batch=1, it is 88% of
// the whole solve, and moves its ~134 MB per panel at roughly 1/48th of the
// device's bandwidth. cuSOLVER has no such cliff.
//
// MEASUREMENT CORRECTION. An earlier version of this comment carried one table
// labelled as covering both modes. It did not: `--name=BM_SYEVX_Crossover` is a
// SUBSTRING filter, so it also ran BM_SYEVX_CrossoverVectors, and the collector
// keyed rows by (n, batch) alone -- the eigenvector rows silently overwrote the
// eigenvalues-only ones. The numbers below are re-measured with the benchmark
// name matched exactly, and the two modes genuinely differ. If you extend this
// grid, filter on the `name` column, not on the --name flag alone.
//
// MEASURED, RTX 4090, CUDA backend, float. Ratio is blocked/vendor, so > 1 means
// vendor wins. WITH EIGENVECTORS:
//
//   n \ batch    1      8     16     32     64    128    256    512   1024
//     64       2.06   4.38   4.44     -    3.93     -    2.28     -      -
//     256      4.31   2.68   2.61     -    2.19   1.84   1.27   1.22     -
//     320        -      -      -    1.34   1.06   0.79   0.69     -      -
//     512      6.45   2.07   1.91   1.62   1.28   0.86   0.74   0.74   0.72
//     640        -      -      -    1.66   1.19   0.85   0.86     -      -
//     896        -      -      -    1.78   1.18   1.19   1.24     -      -
//     1024    15.33   3.33   2.78     -    1.35   1.44   1.46   1.46   1.51
//
// Vendor wins everywhere except one connected box: 320 <= n <= 640 with batch
// >= 128, where the blocked path is ahead by up to 1.37x. Outside it the vendor
// margin reaches 15.4x. The carve-out is stated as the measurement found it
// rather than smoothed into a formula, because it is not monotone in n -- at
// n = 256 and again from n = 768 the vendor wins at every batch measured.
//
// n <= 32 is deliberately excluded: that is CTA territory and the CTA predicate
// is checked first in the order loop. This grid did not measure it.
//
// The grid is float. It is applied to every scalar type because the mechanism is
// the work-group-per-matrix decomposition of the panel kernel, which is identical
// for all of them -- but the carve-out box in particular is a narrow margin
// (1.16-1.37x) and could sit elsewhere in double or complex. Re-measure before
// relying on it there.
//
// EIGENVALUES-ONLY is a different problem and gets a different answer, because
// the two-stage path's stage-2 chase was fixed (it used to run the ~5x slower
// Givens chase in this mode; see two_stage_common.hh). With that fixed, our
// two-stage solver is now the fastest thing available for large n at large batch.
// Measured us/matrix, eigenvalues only:
//
//    n   batch   blocked  two_stage    vendor    two_stage vs vendor
//   256    256      37.8       48.3      27.3    0.57x
//   256   1024      28.1       30.0      27.5    0.92x
//   512    256     266.8      209.6     423.1    2.02x
//   512   1024     289.0      161.4     461.1    2.86x
//  1024    256    3408.5     1110.3    2505.2    2.26x
//  1024   1024    3603.1     1156.6    2626.3    2.27x
//
// So: n >= 512 and batch >= 256, eigenvalues only -> TwoStage, by 2.0-2.9x. At
// n = 256 the vendor still wins at every batch, which is why the gate is on n as
// well as batch.
//
// Below that batch the vendor still wins everywhere, and by a lot at batch 1
// (12.5x at n=1024). That is the same starvation defect again -- both our
// reductions are parallel over the batch only -- and it is the thing to fix if
// these thresholds are ever to move left.
inline bool syev_prefer_vendor(const DeviceCaps& caps, int64_t n, int64_t batch) {
    if (!caps.is_gpu) return false;
    if (n <= 32) return false;
    if (n >= 320 && n <= 640 && batch >= 128) return false;
    return true;
}

// Eigenvalues-only: is the two-stage solver the best choice for this shape?
// Checked before syev_prefer_vendor, since it overrides it where it applies.
inline bool syev_prefer_two_stage_values(const DeviceCaps& caps, int64_t n, int64_t batch) {
    if (!caps.is_gpu) return false;
    return n >= 512 && batch >= 256;
}

template <Backend B, typename T>
inline Provider choose_syev_provider(const DispatchPolicy& policy,
                                     const DeviceCaps& caps,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     Uplo uplo,
                                     JobType jobtype) {
    Provider chosen = normalize_vendor_like(policy.forced);
    // If the user requested a specific provider, try it first. If it cannot support
    // the current matrix/problem (e.g. CTA for n>32), fall back to the regular order
    // instead of hard-failing.
    if (chosen != Provider::Auto) {
        if (chosen == Provider::BatchLAS_CTA && syev_supports_cta(caps, A)) return chosen;
        if (chosen == Provider::BatchLAS_Blocked && syev_supports_blocked(caps, A, uplo)) return chosen;
        if (chosen == Provider::BatchLAS_TwoStage && syev_supports_two_stage(caps, A, uplo)) return chosen;
        if (chosen == Provider::Vendor) return Provider::Vendor;
        // Unsupported request: fall through to Auto selection.
        chosen = Provider::Auto;
    }

    // Auto, and the shape is one the vendor wins: skip the order below, which
    // would otherwise hand every GPU matrix to the blocked path. Restricted to
    // CUDA because that is the only backend the grid above was measured on --
    // the same call on rocSOLVER keeps the historical ordering until someone
    // measures it there.
    if constexpr (B == Backend::CUDA) {
        // Eigenvalues-only at large n and large batch: our two-stage solver wins
        // outright since its stage-2 chase was fixed. Checked first because it
        // overlaps the vendor-preferred region.
        if (jobtype != JobType::EigenVectors &&
            syev_prefer_two_stage_values(caps, A.rows(), A.batch_size()) &&
            syev_supports_two_stage(caps, A, uplo)) {
            return Provider::BatchLAS_TwoStage;
        }
        if (syev_prefer_vendor(caps, A.rows(), A.batch_size())) return Provider::Vendor;
    }

    for (Provider p : policy.order) {
        p = normalize_vendor_like(p);
        if (p == Provider::BatchLAS_CTA && syev_supports_cta(caps, A)) return p;
        if (p == Provider::BatchLAS_Blocked && syev_supports_blocked(caps, A, uplo)) return p;
        if (p == Provider::BatchLAS_TwoStage && syev_supports_two_stage(caps, A, uplo)) return p;
        if (p == Provider::Vendor) return Provider::Vendor;
    }

    return Provider::Vendor;
}

} // namespace detail

// Backend-agnostic provider selection + orchestration.
// Actual vendor calls are provided by `backend::syev_vendor`.
template <Backend B, typename T>
inline Event syev_dispatch(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& descrA,
                           Span<typename base_type<T>::type> eigenvalues,
                           JobType jobtype,
                           Uplo uplo,
                           Span<std::byte> workspace) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("SYEV");
    Provider chosen = detail::choose_syev_provider<B, T>(policy, caps, descrA, uplo, jobtype);

    if constexpr (B == Backend::NETLIB) {
        chosen = Provider::Vendor;
    }

    size_t need_ws = 0;
    if (chosen == Provider::Vendor) {
        need_ws = backend::syev_vendor_buffer_size<B, T>(ctx, descrA, eigenvalues, jobtype, uplo);
    } else if (chosen == Provider::BatchLAS_CTA) {
        need_ws = syev_cta_buffer_size<B, T>(ctx, descrA, jobtype, detail::syev_cta_steqr_params<T>(jobtype));
    } else if (chosen == Provider::BatchLAS_TwoStage) {
        need_ws = syev_two_stage_buffer_size<B, T>(ctx,
                                                   descrA,
                                                   jobtype,
                                                   uplo,
                                                   StedcParams<typename base_type<T>::type>{});
    } else if (chosen == Provider::BatchLAS_Blocked) {
        need_ws = syev_blocked_buffer_size<B, T>(ctx,
                                                 descrA,
                                                 jobtype,
                                                 uplo,
                                                 StedcParams<typename base_type<T>::type>{});
    } else {
        chosen = Provider::Vendor;
        need_ws = backend::syev_vendor_buffer_size<B, T>(ctx, descrA, eigenvalues, jobtype, uplo);
    }

    if (workspace.size() < need_ws) {
        throw std::runtime_error("syev: insufficient workspace for chosen provider");
    }

    Queue* run_q = &ctx;
    if (!ctx.in_order()) {
        Queue in_order_q(run_q->device(), true);
        in_order_q = Queue(ctx, true);
        Event dep = ctx.get_event();
        in_order_q.enqueue(dep);
        run_q = &in_order_q;
    }

    Event e;
    if (chosen == Provider::Vendor) {
        e = backend::syev_vendor<B, T>(*run_q, descrA, eigenvalues, jobtype, uplo, workspace);
    } else if (chosen == Provider::BatchLAS_CTA) {
        e = syev_cta<B, T>(*run_q,
                           descrA,
                           eigenvalues,
                           jobtype,
                           uplo,
                           workspace,
                           detail::syev_cta_steqr_params<T>(jobtype),
                           /*cta_wg_size_multiplier=*/1);
    } else if (chosen == Provider::BatchLAS_TwoStage) {
        e = syev_two_stage<B, T>(*run_q,
                                 descrA,
                                 eigenvalues,
                                 jobtype,
                                 uplo,
                                 workspace,
                                 StedcParams<typename base_type<T>::type>{});
    } else {
        e = syev_blocked<B, T>(*run_q,
                               descrA,
                               eigenvalues,
                               jobtype,
                               uplo,
                               workspace,
                               StedcParams<typename base_type<T>::type>{});
    }

    return e;
}

template <Backend B, typename T>
inline size_t syev_buffer_size_dispatch(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& descrA,
                                        Span<typename base_type<T>::type> eigenvalues,
                                        JobType jobtype,
                                        Uplo uplo) {
    const DeviceCaps caps = query_caps(ctx);
    const DispatchPolicy policy = policy_from_env("SYEV");
    Provider chosen = detail::choose_syev_provider<B, T>(policy, caps, descrA, uplo, jobtype);

    if constexpr (B == Backend::NETLIB) {
        chosen = Provider::Vendor;
    }

    if (chosen == Provider::Vendor) {
        return backend::syev_vendor_buffer_size<B, T>(ctx, descrA, eigenvalues, jobtype, uplo);
    }
    if (chosen == Provider::BatchLAS_CTA) {
        return syev_cta_buffer_size<B, T>(ctx, descrA, jobtype, detail::syev_cta_steqr_params<T>(jobtype));
    }
    if (chosen == Provider::BatchLAS_TwoStage) {
        return syev_two_stage_buffer_size<B, T>(ctx,
                                                descrA,
                                                jobtype,
                                                uplo,
                                                StedcParams<typename base_type<T>::type>{});
    }
    return syev_blocked_buffer_size<B, T>(ctx,
                                          descrA,
                                          jobtype,
                                          uplo,
                                          StedcParams<typename base_type<T>::type>{});
}

} // namespace batchlas::blas::dispatch

namespace batchlas {

template <Backend B, typename T>
inline Event syev(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& descrA,
                  Span<typename base_type<T>::type> eigenvalues,
                  JobType jobtype,
                  Uplo uplo,
                  Span<std::byte> workspace) {
    return blas::dispatch::syev_dispatch<B, T>(ctx, descrA, eigenvalues, jobtype, uplo, workspace);
}

template <Backend B, typename T>
inline size_t syev_buffer_size(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& descrA,
                               Span<typename base_type<T>::type> eigenvalues,
                               JobType jobtype,
                               Uplo uplo) {
    return blas::dispatch::syev_buffer_size_dispatch<B, T>(ctx, descrA, eigenvalues, jobtype, uplo);
}

} // namespace batchlas

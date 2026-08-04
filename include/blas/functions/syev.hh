#pragma once

#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <string_view>

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

// Uplo::Upper is supported: syev_blocked mirrors the upper triangle into the lower one
// (an O(n^2) pass, see src/extensions/uplo_mirror.hh) and runs the ordinary Lower pipeline.
// Before that, every Upper call fell through to the vendor no matter how much faster this
// path was at that shape.
template <typename T>
inline bool syev_supports_blocked(const DeviceCaps& caps,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  Uplo uplo) {
    (void)uplo;
    if (A.rows() != A.cols()) return false;
    if (A.rows() < 1 || A.batch_size() < 1) return false;
    if (!caps.is_gpu) return false;
    return true;
}

// Uplo::Upper supported by the same mirror; see syev_supports_blocked above.
template <typename T>
inline bool syev_supports_two_stage(const DeviceCaps& caps,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    Uplo uplo) {
    (void)uplo;
    if (A.rows() != A.cols()) return false;
    if (A.rows() < 1 || A.batch_size() < 1) return false;
    if (!caps.is_gpu) return false;
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
// RE-MEASURED 2026-08-04 (RTX 4090 device 1, build 12963a8, float, eigenvectors,
// median of 3, one process at a time). The grid above was taken at 27851a6, i.e.
// BEFORE the grid-barrier latrd path (87f6887, defaulted on at n >= 768 by
// 5401f63) existed, so it was due a re-check. blocked/vendor, > 1 = vendor wins:
//
//   n \ batch     1     8    16    32    64   128   256   512  1024
//     64       2.08  4.43  4.48  4.32  3.93  3.22  2.29  1.55  1.11
//     128      2.00  4.16  4.09  3.79  3.39  2.71  1.72  1.22  0.98
//     256      3.96  2.67  2.59  2.44  2.19  1.87  1.26  1.21  1.08
//     320      5.00  1.23  1.35  1.24  1.03  0.79  0.68  0.47  0.36
//     512      6.42  2.07  1.93  1.62  1.27  0.86  0.73  0.74     -
//     640      8.78  2.27  2.03  1.67  1.18  0.85  0.86  0.81     -
//     768      9.99  2.56  2.21  1.73  1.10  1.05  1.07     -     -
//     896     10.05  2.27  1.93  1.52  1.06  1.19  1.24     -     -
//     1024    10.14  2.29  1.97  1.50  1.15  1.44     -     -     -
//
// SUPERSEDED 2026-08-04 by the saturated sweep in syev_saturated_provider_for_n below.
// THIS GRID IS RETAINED ONLY AS A RECORD OF THE BATCH-DEPENDENT BEHAVIOUR; it no longer
// drives eigenvector routing. Its batch ladder stopped at 1024, which for small n is NOWHERE
// NEAR saturation -- at n = 64 the true crossover needs batch ~16384, and there blocked wins
// by 1.23x where this grid shows the vendor ahead at 1.11x. Reading routing off it was wrong
// at five of nine sizes. Do not reinstate a batch-keyed rule from these numbers.
//
// The original conclusion, now known to be an artifact of the unsaturated ladder:
//   * n = 1024 improved a lot -- 15.33 -> 10.14 at batch 1, 3.33 -> 2.29 at
//     batch 8 -- which is grid-latrd doing exactly its job. The vendor still
//     wins there, so the decision does not flip; only the margin narrowed.
//   * The carve-out is WIDER than recorded, because the original grid stopped at
//     batch 256. The blocked win keeps growing with batch: at n = 320 it reaches
//     0.47 at batch 512 and 0.36 at batch 1024, i.e. blocked is 2.8x faster than
//     cuSOLVER there. Same direction at n = 512 (0.74 at batch 512) and n = 640
//     (0.81 at batch 512). The `batch >= 128` predicate already covers all of it.
//   * n = 128 at batch 1024 came in at 0.98 -- nominally ours, but 1.02x is
//     inside the noise band and is NOT carved out.
//   * n = 768 and 896 are vendor-win at every batch, confirming the upper edge.
//
// A CONTENTION WARNING, because it nearly produced a wrong table. A first pass
// had n=768 at 0.28 and 0.38 for batch 32 and 64 -- an apparent 3.6x blocked win
// that would have moved the carve-out's upper edge. It was an artifact: two
// measuring processes overlapped on the device, and the VENDOR arm (not blocked)
// was inflated to 6885 us/matrix against 1110 when re-run alone. Re-measured with
// the GPU to itself, the row is 1.73 / 1.10 / 1.05 -- vendor wins throughout.
// Never measure this grid with anything else running on the device.
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

// --- Small n: where does the vendor overtake the CTA solver? ---------------
//
// syev_supports_cta claims *every* n <= 32, and the Auto order checks it first,
// so every small dense eigensolve on a GPU lands on CTA regardless of cost. That
// is not free. The projected Rayleigh-Ritz solve inside LOBPCG and the filtered
// solver (dimension up to 3 * block_vectors, so <= 32 for the usual block sizes)
// is the case that exposed it:
//
//   n = 30, batch = 8, float, eigenvectors:  CTA 229.6 us/call
//                                            cuSOLVER 103.7 us/call   (2.21x)
//
// and nsys attributed 29.4% of all LOBPCG GPU time to that solve, i.e. roughly
// 16% end-to-end. syev_prefer_vendor above cannot fix this: it returns false for
// n <= 32 by construction.
//
// HONESTY ABOUT WHAT IS MEASURED. Exactly one point in this range was measured:
// n = 30, batch = 8, float, with eigenvectors. The crossover below it was NOT
// measured. CTA is expected to win at very small n, where a vendor launch costs
// more than the whole problem, so the threshold is a real one and not just "never
// use CTA" -- but its value is a guess, deliberately exposed as a single knob:
//
//   BATCHLAS_SYEV_CTA_MAX_N=<n>   Auto uses CTA only for n <= this (default 16).
//                                 Set 32 to restore the previous behaviour
//                                 exactly (CTA claims the whole range); set 0 to
//                                 send every small eigenvector solve to the
//                                 vendor. Sweeping it over 0..32 is how the
//                                 crossover should be found.
//
// Scope is kept deliberately narrow:
//   * CUDA only -- the 2.21x is a cuSOLVER number and nothing else was measured;
//   * eigenvector mode only -- eigenvalues-only at n <= 32 was not measured, so
//     it keeps CTA and cannot regress;
//   * only where CTA would actually have been chosen, so nothing that already
//     routes elsewhere changes;
//   * an explicitly forced Provider::BatchLAS_CTA still wins, since the forced
//     branch returns before this is consulted.
inline int64_t syev_cta_max_n_for_vectors() {
    // Default 32 == OFF: CTA keeps the whole n <= 32 range, as it always has.
    //
    // Lowering this IS a measured speedup for the projected Rayleigh-Ritz solve
    // inside LOBPCG. Measured, LOBPCG EigenVectors, n=256, us/matrix:
    //
    //   BATCHLAS_SYEV_CTA_MAX_N     32(off)    16      8       0
    //   batch 8                      15563   14211  13746   13551
    //   batch 64                      1998.6  1890.4 1948.0  1945.9
    //
    // i.e. 1.10-1.15x at batch 8 and 1.03-1.06x at batch 64, monotone in the
    // threshold. It is nevertheless OFF by default because moving the projected
    // solve off CTA perturbs LOBPCG's numerics just enough to flip a marginal case
    // in ILUKTests.SyevxInstrumentationAndPreconditioner, which asserts that ILU(k)
    // beats the unpreconditioned baseline on EVERY case (lose_count == 0). At
    // threshold 16 one of eight cases (d0.06_b0.5_s1234, iluk_k2) crosses to
    // ratio 1.25 -- at a point where the baseline has already converged to 4.2e-06,
    // so it is a near-tie rather than a real degradation, but it is a real test
    // failure and this is someone else's correctness assertion, not mine to relax.
    //
    // Resolving it means deciding whether that assertion should tolerate a tie on an
    // already-converged case, or whether the vendor projected solve genuinely hurts
    // LOBPCG convergence. Until then, a 1.1x win does not justify shipping a red test.
    constexpr int64_t kDefault = 32;
    const char* v = std::getenv("BATCHLAS_SYEV_CTA_MAX_N");
    if (!v || !*v) return kDefault;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed < 0 || parsed > 32) return kDefault;
    return static_cast<int64_t>(parsed);
}

// --- Which small-n kernel? -------------------------------------------------
//
// `Auto` used to send EVERY n <= 32 to syev_cta. Measured on an RTX 4090 (2026-08-03, build
// 7911847, device 1, median of 5 repeats with IQR, each at that cell's measured knee batch),
// syev_cta does not win a single cell in either precision -- and the two kernels that beat it
// were both unreachable from `Auto`. us/matrix, best over cta_wg_size_multiplier:
//
//   type   n  mode     winner      vs syev_cta   vs cuSOLVER
//   double 4  values   jacobi          3.87x         --
//   double 4  vectors  jacobi          3.75x       15.17x
//   double 8  vectors  jacobi          2.91x        3.70x
//   double 16 vectors  jacobi          2.50x        3.29x
//   double 32 vectors  jacobi           --          1.37x
//   float  4  vectors  jacobi          4.55x       18.57x
//   float  8  vectors  jacobi          2.55x        8.42x
//   float  16 vectors  cta_fused       1.25x        4.10x
//   float  32 vectors  cta_fused       1.12x        1.79x
//
// Full tables, provenance and the two cells that came out NEUTRAL (float n=16 and n=32,
// eigenvalues-only) are in SYEV_RETUNE_RESULTS.md.
//
// SCOPE -- deliberately only what was measured:
//   * REAL types only. Complex was not measured and keeps syev_cta.
//   * n <= 32 only, i.e. exactly the range syev_cta already claimed.
//   * Uplo::Lower is what the benchmarks exercise. Both kernels accept either; cta_fused
//     internally runs the Upper path and transposes Lower, so Lower is the costlier case and
//     is the one measured.
//
// FP64 CAVEAT, load-bearing. This GPU runs FP64 at 1/64 rate, which inflates Jacobi's margin
// over the tridiagonalizing path. The float column is the better predictor for a 1:2 FP64
// datacenter GPU. The `double` rule below should be re-measured there before being trusted;
// BATCHLAS_SYEV_SMALL_KERNEL=cta restores the previous behaviour wholesale.
//
// Jacobi additionally has a large relative-accuracy advantage on graded SPD input
// (4.5e-07 vs syev_cta's 2.7e+28, JACOBI_EIGENSOLVER_PLAN.md 13.1). That is a reason to
// prefer it on ties; it is not the reason it is routed here -- it also wins on speed.
enum class SyevSmallKernel { Cta, CtaFused, Jacobi };

inline SyevSmallKernel syev_small_kernel_env(bool& forced) {
    forced = true;
    const char* v = std::getenv("BATCHLAS_SYEV_SMALL_KERNEL");
    if (v && *v) {
        const std::string_view s(v);
        if (s == "cta") return SyevSmallKernel::Cta;
        if (s == "fused" || s == "cta_fused") return SyevSmallKernel::CtaFused;
        if (s == "jacobi") return SyevSmallKernel::Jacobi;
    }
    forced = false;
    return SyevSmallKernel::Cta;
}

template <typename T>
inline SyevSmallKernel syev_choose_small_kernel(const MatrixView<T, MatrixFormat::Dense>& A) {
    bool forced = false;
    const SyevSmallKernel env = syev_small_kernel_env(forced);
    if (forced) return env;

    // Complex was never measured -- keep the historical kernel. `internal::is_complex` lives
    // in src/math-helpers.hh and is not visible from this public header, so detect complex
    // via the public base_type trait: for a real T, base_type<T>::type IS T.
    constexpr bool kReal = std::is_same_v<T, typename base_type<T>::type>;
    if constexpr (!kReal) {
        return SyevSmallKernel::Cta;
    } else {
        constexpr bool is_double = std::is_same_v<typename base_type<T>::type, double>;
        if constexpr (is_double) {
            return SyevSmallKernel::Jacobi;      // wins at every measured n <= 32
        } else {
            return A.rows() <= 8 ? SyevSmallKernel::Jacobi   // 2.2x - 4.6x
                                 : SyevSmallKernel::CtaFused; // 1.03x - 1.25x
        }
    }
}

template <typename T>
inline bool syev_prefer_vendor_over_cta(const DeviceCaps& caps,
                                        const MatrixView<T, MatrixFormat::Dense>& A,
                                        JobType jobtype) {
    if (!caps.is_gpu) return false;
    if (jobtype != JobType::EigenVectors) return false;
    if (!syev_supports_cta(caps, A)) return false;
    return A.rows() > syev_cta_max_n_for_vectors();
}

// Eigenvalues-only: is the two-stage solver the best choice for this shape?
// Checked before syev_prefer_vendor, since it overrides it where it applies.
// SUPERSEDED by syev_saturated_provider_for_n_values. Retained only because the batch >= 256
// term is a documented cautionary tale: an n = 1024 solve at batch 254 fell through it to the
// vendor and paid 2.75x. No longer consulted by choose_syev_provider.
inline bool syev_prefer_two_stage_values(const DeviceCaps& caps, int64_t n, int64_t batch) {
    if (!caps.is_gpu) return false;
    return n >= 512 && batch >= 256;
}

// --- Eigenvector routing, decided AT SATURATION and keyed on n alone ----------
//
// WHY THIS REPLACES syev_prefer_vendor FOR EIGENVECTORS.
//
// Every earlier grid in this file capped its batch ladder at 1024, and for small n that is
// nowhere near saturation -- at n = 64 the vendor's lower fixed cost still dominates there,
// so the vendor looked like the winner. Measured at a batch large enough to amortise launch
// overhead, it is not. Routing is a per-n decision made on the assumption that the batch is
// large; a caller running tiny batches pays launch overhead whatever we pick, and tuning the
// routing for that regime costs the saturated regime real throughput.
//
// MEASURED 2026-08-04, RTX 4090 device 1, build 12963a8, float, EIGENVECTORS, us/matrix,
// median of 3, one process on the device. Batch per n is the largest at which ALL THREE
// providers fit (blocked and two-stage carry much larger workspaces than the vendor path):
//
//     n   batch    blocked    vendor  two_stage    winner        margin
//    64   16384       1.64      2.02       2.21    blocked        1.23x
//   128    4069       6.65      7.85      10.35    blocked        1.18x
//   256    2034      36.92     37.14      57.56    blocked        1.01x  (tie w/ vendor)
//   320    1302      74.53    215.23     111.20    blocked        1.49x
//   512     508     384.27    504.69     380.02    two_stage      1.01x  (tie w/ blocked)
//   640     651     836.30   1008.06     727.48    two_stage      1.15x
//   768     452    1612.98   1414.99    1184.11    two_stage      1.19x
//  1024     254    4089.02   2706.84    2441.93    two_stage      1.11x
//  2048      64   33842.60  15019.10   24782.30    vendor         1.65x
//
// The old routing sent 64, 128, 768 and 1024 to the vendor and 640 to blocked -- five of the
// nine rows wrong, by 1.11x to 1.23x.
//
// n <= 32 is NOT handled here: it is CTA territory and syev_choose_small_kernel picks among
// the three CTA kernels. Measured at batch 2048, cta/jacobi (n=16) and cta/fused (n=32) beat
// every non-CTA provider by 2.8x-3.0x, so the CTA-first ordering is correct there.
//
// CAVEATS, all load-bearing:
//   * n = 2048 was measured at batch 64, which is BELOW the 128 SMs and therefore NOT
//     saturated -- it is memory-limited, since blocked/two-stage cannot fit a larger batch at
//     that size. Its "vendor" verdict is the weakest row in the table and should be
//     re-measured on a card with more memory before being relied on.
//   * float only. The mechanism (work-per-matrix vs launch overhead) is type-independent, but
//     the crossovers could sit elsewhere in double or complex.
//   * EIGENVECTORS only. Eigenvalues-only has its own table and its own rule; see
//     syev_saturated_provider_for_n_values below. (syev_benchmark grew a jobz argument so
//     that mode could be measured against the vendor at all.)
inline Provider syev_saturated_provider_for_n(int64_t n) {
    if (n <= 320) return Provider::BatchLAS_Blocked;   // 64..320, up to 1.49x over the vendor
    if (n <= 1024) return Provider::BatchLAS_TwoStage; // 512..1024, up to 1.19x
    return Provider::Vendor;                           // 2048+, 1.65x (but see the caveat)
}

// --- The same, for EIGENVALUES-ONLY -----------------------------------------
//
// Measured 2026-08-04, same method and machine, jobz = NoEigenVectors. This mode could only
// be measured once syev_benchmark grew a jobz argument -- it previously hardcoded
// JobType::EigenVectors, so there was no vendor arm to compare against.
//
//     n   batch    blocked    vendor  two_stage    winner      margin
//    64   16384       1.06      1.12       1.14    blocked      1.06x
//   128    4069       4.41      5.34       5.35    blocked      1.21x
//   256    2034      24.78     27.20      27.98    blocked      1.10x
//   320    2604      48.16    200.94      50.35    blocked      1.05x
//   512    1017     298.30    458.95     158.72    two_stage    1.88x
//   640     651     687.67    929.05     330.71    two_stage    2.08x
//   768     452    1362.46   1301.59     505.29    two_stage    2.58x
//  1024     254    3537.41   2494.44     908.32    two_stage    2.75x
//  2048      64   29547.70  13908.80   10804.10    two_stage    1.29x
//
// Same boundaries as the eigenvector table for 64..320, but two-stage keeps winning all the
// way to 2048 rather than handing over to the vendor -- and by far larger margins (1.88x to
// 2.75x, against at most 1.19x with eigenvectors). That is the back-transform: with no
// eigenvectors to produce, two-stage keeps its cheap band reduction and never pays to apply
// Q2, so its flop advantage survives into wall-clock.
//
// THIS REPLACES syev_prefer_two_stage_values, WHOSE BATCH TERM DID REAL DAMAGE. That
// predicate required batch >= 256; the n = 1024 cell above ran at batch 254 and therefore
// fell through to the vendor and paid 2.75x. Two callers differing by two matrices got
// completely different providers for reasons unrelated to which kernel is better. Routing is
// keyed on n alone.
inline Provider syev_saturated_provider_for_n_values(int64_t n) {
    if (n <= 320) return Provider::BatchLAS_Blocked;   // 64..320, 1.05x - 1.21x
    return Provider::BatchLAS_TwoStage;                // 512..2048, 1.29x - 2.75x
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
        // EIGENVALUES-ONLY, n > 32: routed per n from the saturated measurement, replacing
        // syev_prefer_two_stage_values. Its batch >= 256 term sent an n=1024 solve at batch
        // 254 to the vendor at 2.75x the cost of two-stage; routing is keyed on n alone.
        if (jobtype != JobType::EigenVectors && A.rows() > 32) {
            const Provider want = syev_saturated_provider_for_n_values(A.rows());
            if (want == Provider::BatchLAS_Blocked && syev_supports_blocked(caps, A, uplo)) {
                return want;
            }
            if (want == Provider::BatchLAS_TwoStage && syev_supports_two_stage(caps, A, uplo)) {
                return want;
            }
            return Provider::Vendor;
        }
        // EIGENVECTORS, n > 32: routed per n from the saturated measurement above, NOT from
        // the batch-dependent syev_prefer_vendor grid, which was built from unsaturated
        // cells and picks the wrong provider at five of nine measured sizes. n <= 32 falls
        // through to the order loop so the CTA branch keeps it.
        if (jobtype == JobType::EigenVectors && A.rows() > 32) {
            const Provider want = syev_saturated_provider_for_n(A.rows());
            if (want == Provider::BatchLAS_Blocked && syev_supports_blocked(caps, A, uplo)) {
                return want;
            }
            if (want == Provider::BatchLAS_TwoStage && syev_supports_two_stage(caps, A, uplo)) {
                return want;
            }
            // Unsupported for this shape -- the vendor handles everything. (Uplo::Upper is
            // no longer such a case; both BatchLAS paths mirror it into Lower.)
            return Provider::Vendor;
        }
        // Only reachable for n <= 32 now, where it returns false by construction -- both
        // jobz branches above return for n > 32. Kept so an explicit provider order still
        // behaves, but it no longer decides anything. See the SUPERSEDED note on the grid.
        if (syev_prefer_vendor(caps, A.rows(), A.batch_size())) return Provider::Vendor;
        // Small n with eigenvectors: CTA claims the whole n <= 32 range, but the
        // vendor is faster over part of it. See syev_prefer_vendor_over_cta.
        if (syev_prefer_vendor_over_cta<T>(caps, A, jobtype)) return Provider::Vendor;
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
        switch (detail::syev_choose_small_kernel<T>(descrA)) {
            case detail::SyevSmallKernel::Jacobi:
                need_ws = syev_jacobi_cta_buffer_size<B, T>(ctx, descrA, jobtype);
                break;
            case detail::SyevSmallKernel::CtaFused:
                need_ws = syev_cta_fused_buffer_size<B, T>(ctx, descrA, jobtype,
                                                           detail::syev_cta_steqr_params<T>(jobtype));
                break;
            default:
                need_ws = syev_cta_buffer_size<B, T>(ctx, descrA, jobtype,
                                                     detail::syev_cta_steqr_params<T>(jobtype));
                break;
        }
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
        // Which of the three n<=32 kernels -- see syev_choose_small_kernel. The workspace
        // query above MUST take the same branch; both call the same selector, and the
        // selector reads its env override fresh, so it must not be flipped between the
        // buffer-size query and the call.
        switch (detail::syev_choose_small_kernel<T>(descrA)) {
            case detail::SyevSmallKernel::Jacobi:
                e = syev_jacobi_cta<B, T>(*run_q, descrA, eigenvalues, jobtype, uplo, workspace);
                break;
            case detail::SyevSmallKernel::CtaFused:
                e = syev_cta_fused<B, T>(*run_q,
                                         descrA,
                                         eigenvalues,
                                         jobtype,
                                         uplo,
                                         workspace,
                                         detail::syev_cta_steqr_params<T>(jobtype),
                                         /*cta_wg_size_multiplier=*/1);
                break;
            default:
                e = syev_cta<B, T>(*run_q,
                                   descrA,
                                   eigenvalues,
                                   jobtype,
                                   uplo,
                                   workspace,
                                   detail::syev_cta_steqr_params<T>(jobtype),
                                   /*cta_wg_size_multiplier=*/1);
                break;
        }
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
        // Must mirror syev_dispatch exactly: a caller that sizes its workspace here and then
        // runs a different small-n kernel would under-allocate. Both sites call the same
        // selector with the same arguments.
        switch (detail::syev_choose_small_kernel<T>(descrA)) {
            case detail::SyevSmallKernel::Jacobi:
                return syev_jacobi_cta_buffer_size<B, T>(ctx, descrA, jobtype);
            case detail::SyevSmallKernel::CtaFused:
                return syev_cta_fused_buffer_size<B, T>(ctx, descrA, jobtype,
                                                        detail::syev_cta_steqr_params<T>(jobtype));
            default:
                return syev_cta_buffer_size<B, T>(ctx, descrA, jobtype,
                                                  detail::syev_cta_steqr_params<T>(jobtype));
        }
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

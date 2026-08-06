// Relative-accuracy instrument for batched SVD, swept over conditioning.
//
// This exists because the pre-existing gesvd accuracy benchmarks cannot see the
// effect that decides the comparison against cuSOLVER's gesvdjBatched
// (GESVD_PLAN.md defect C). Two independent reasons they are blind:
//
//   1. They report an ABSOLUTE singular-value error (max_abs_singular_error).
//      The normal-equations defect shows up as a loss of RELATIVE accuracy in
//      the SMALL singular values; an absolute metric normalised by sigma_max
//      cannot express it.
//   2. They never call state.target_log10_cond() and never record a metric named
//      "log10_cond", so the --log10-cond sweep does nothing and the log10cond
//      column prints blank.
//
// Both are fixed here. Also note miniacc's terminal summary prints only a fixed
// whitelist of metric names (miniacc.hh:425), which is the other reason the old
// metrics were invisible -- the names below (R, O, max_relerr) are chosen to be
// on that whitelist so the numbers actually reach the terminal, not just --csv.
//
// The reference is exact, not another solve: random_with_log10_cond_metric
// builds A = U * S * V^H from an explicitly constructed geometric spectrum
// (src/extra/random_cond.cc:46), so the true singular values are known in closed
// form. Comparing against a second floating-point solve would put the reference
// at the same error floor as the thing being measured.

#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <blas/functions.hh>
#include <util/miniacc.hh>

#include "acc_utils.hh"
#include "miniacc_accuracy_common.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename Benchmark>
void GesvdRelAccSizes(Benchmark* b) {
    for (double n : {8.0, 16.0, 32.0}) b->Args({n});
}

// The spectrum random_with_log10_cond_metric actually builds, in closed form:
//   s_i = exp((i - (n-1)/2) * ln(10) * log10_kappa2 / (n-1)),  i ascending.
// (src/extra/random_cond.cc:46). Returned DESCENDING to match the BatchLAS gesvd
// output convention -- which is descending, produced by an index reversal in
// finalize_values_only, not by a sort flag.
std::vector<double> reference_spectrum_descending(int n, double log10_kappa2) {
    std::vector<double> s(static_cast<size_t>(n));
    if (n == 1) {
        s[0] = 1.0;
        return s;
    }
    const double ln10 = std::log(10.0);
    for (int i = 0; i < n; ++i) {
        const double e = (static_cast<double>(i) - 0.5 * (n - 1)) * ln10 * log10_kappa2 / (n - 1);
        s[static_cast<size_t>(i)] = std::exp(e);
    }
    std::reverse(s.begin(), s.end());
    return s;
}

struct SvdErrors {
    double max_relerr;   // max_i |sigma_i - sigma_i_ref| / sigma_i_ref
    double recon;        // ||A - U S V^H||_F / ||A||_F
    double ortho;        // max(||U^H U - I||_max, ||V^H V - I||_max)
};

// Host-side verification. n <= 32 here, so an O(n^3) host check per sample is
// cheap and is the trustworthy choice: it shares no code with the kernel under
// test.
template <typename Real>
SvdErrors svd_errors_host(int n,
                          const Real* A,      // n x n col-major, ld = n (pristine copy)
                          const Real* U,      // n x n col-major, ld = n
                          const Real* Vh,     // n x n col-major, ld = n  (already V^H)
                          const Real* s,
                          const std::vector<double>& s_ref) {
    SvdErrors e{0.0, 0.0, 0.0};

    for (int i = 0; i < n; ++i) {
        const double got = static_cast<double>(s[i]);
        const double ref = s_ref[static_cast<size_t>(i)];
        if (ref > 0.0) {
            e.max_relerr = std::max(e.max_relerr, std::abs(got - ref) / ref);
        }
    }

    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double acc = 0.0;
            for (int t = 0; t < n; ++t) {
                // (U S V^H)_ij = sum_t U(i,t) * s_t * Vh(t,j)
                acc += static_cast<double>(U[static_cast<size_t>(t) * n + i]) *
                       static_cast<double>(s[t]) *
                       static_cast<double>(Vh[static_cast<size_t>(j) * n + t]);
            }
            const double a = static_cast<double>(A[static_cast<size_t>(j) * n + i]);
            num += (a - acc) * (a - acc);
            den += a * a;
        }
    }
    e.recon = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);

    auto gram_defect = [n](const Real* M, bool rows_are_vectors) {
        double worst = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = 0; q < n; ++q) {
                double acc = 0.0;
                for (int t = 0; t < n; ++t) {
                    const double x = rows_are_vectors
                        ? static_cast<double>(M[static_cast<size_t>(t) * n + p])   // row p
                        : static_cast<double>(M[static_cast<size_t>(p) * n + t]);  // col p
                    const double y = rows_are_vectors
                        ? static_cast<double>(M[static_cast<size_t>(t) * n + q])
                        : static_cast<double>(M[static_cast<size_t>(q) * n + t]);
                    acc += x * y;
                }
                worst = std::max(worst, std::abs(acc - (p == q ? 1.0 : 0.0)));
            }
        }
        return worst;
    };

    // U's columns are the left singular vectors; Vh's ROWS are the right ones.
    e.ortho = std::max(gram_defect(U, /*rows_are_vectors=*/false),
                       gram_defect(Vh, /*rows_are_vectors=*/true));
    return e;
}

enum class RelAccImpl { BatchlasCta, CusolverJacobi };

template <typename Real, Backend B, RelAccImpl Impl>
void run_gesvd_relacc(miniacc::State& state) {
    const int n = std::max(2, state.arg_int(0));
    const int chunk_batch = miniacc_acc::chunk_batch_from_samples(state.samples());

    const double target_log10_raw = state.target_log10_cond();
    // Default when --log10-cond is absent: a mild but non-trivial conditioning.
    const double target_log10 = std::isfinite(target_log10_raw) ? target_log10_raw : 1.0;

    // in-order: the native CTA provider requires it.
    auto q = std::make_shared<Queue>(Device("gpu"), B, true);

    state.SetTag("impl", Impl == RelAccImpl::BatchlasCta ? "gesvd_cta" : "cusolver_gesvdj");
    state.SetTag("backend", miniacc_acc::backend_name<B>());
    state.SetTag("dtype", miniacc_acc::dtype_name<Real>());

    const std::vector<double> s_ref = reference_spectrum_descending(n, target_log10);

    size_t produced = 0;
    while (produced < state.samples()) {
        const int cur_batch = static_cast<int>(
            std::min<size_t>(static_cast<size_t>(chunk_batch), state.samples() - produced));
        const unsigned seed = state.seed() + static_cast<unsigned>(produced);

        auto A = random_with_log10_cond_metric<B, Real>(
            *q, n, static_cast<Real>(target_log10), NormType::Spectral, cur_batch, seed);
        q->wait();

        auto A_work = A.clone();
        Matrix<Real> U(n, n, cur_batch);
        Matrix<Real> Vh(n, n, cur_batch);
        UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(cur_batch));

        bool failed = false;
        std::string reason;
        try {
            if constexpr (Impl == RelAccImpl::BatchlasCta) {
                const size_t ws_bytes = gesvd_cta_buffer_size<B, Real>(
                    *q, A_work.view(), s.to_span(), U.view(), Vh.view(),
                    SvdVectors::All, SvdVectors::All);
                UnifiedVector<std::byte> ws(ws_bytes);
                gesvd_cta<B, Real>(*q, A_work.view(), s.to_span(), U.view(), Vh.view(),
                                   SvdVectors::All, SvdVectors::All, ws.to_span());
            } else {
                const size_t ws_bytes = backend::gesvd_vendor_buffer_size<B, Real>(
                    *q, A_work.view(), s.to_span(), U.view(), Vh.view(),
                    SvdVectors::All, SvdVectors::All);
                UnifiedVector<std::byte> ws(ws_bytes);
                backend::gesvd_vendor<B, Real>(*q, A_work.view(), s.to_span(), U.view(), Vh.view(),
                                               SvdVectors::All, SvdVectors::All, ws.to_span());
            }
            q->wait_and_throw();
        } catch (const std::exception& ex) {
            failed = true;
            reason = ex.what();
        }

        if (failed) {
            for (int b = 0; b < cur_batch; ++b) {
                state.RecordSample({{"max_relerr", std::numeric_limits<double>::quiet_NaN()},
                                    {"R", std::numeric_limits<double>::quiet_NaN()},
                                    {"O", std::numeric_limits<double>::quiet_NaN()},
                                    {"log10_cond", target_log10}},
                                   false, reason);
            }
            produced += static_cast<size_t>(cur_batch);
            continue;
        }

        // Read the batch item through the view's own stride rather than assuming
        // n*n packing; Matrix is packed today but the host check should not depend
        // on that.
        const Real* A_base = A.view().data_ptr();
        const Real* U_base = U.view().data_ptr();
        const Real* Vh_base = Vh.view().data_ptr();
        const int64_t A_stride = A.view().stride();
        const int64_t U_stride = U.view().stride();
        const int64_t Vh_stride = Vh.view().stride();

        for (int b = 0; b < cur_batch; ++b) {
            const size_t soff = static_cast<size_t>(b) * static_cast<size_t>(n);
            const SvdErrors e = svd_errors_host<Real>(
                n,
                A_base + static_cast<size_t>(b) * static_cast<size_t>(A_stride),
                U_base + static_cast<size_t>(b) * static_cast<size_t>(U_stride),
                Vh_base + static_cast<size_t>(b) * static_cast<size_t>(Vh_stride),
                s.data() + soff,
                s_ref);

            const bool ok = std::isfinite(e.max_relerr) && std::isfinite(e.recon) && std::isfinite(e.ortho);
            state.RecordSample({{"max_relerr", e.max_relerr},
                                {"R", e.recon},
                                {"O", e.ortho},
                                {"log10_cond", target_log10}},
                               ok,
                               ok ? "" : "non_finite_error");
        }

        produced += static_cast<size_t>(cur_batch);
    }
}

} // namespace

template <typename Real, Backend B>
void ACC_GESVD_RELACC_BATCHLAS(miniacc::State& state) {
    run_gesvd_relacc<Real, B, RelAccImpl::BatchlasCta>(state);
}

template <typename Real, Backend B>
void ACC_GESVD_RELACC_CUSOLVER(miniacc::State& state) {
    run_gesvd_relacc<Real, B, RelAccImpl::CusolverJacobi>(state);
}

BATCHLAS_ACC_CUDA(ACC_GESVD_RELACC_BATCHLAS, GesvdRelAccSizes)
BATCHLAS_ACC_CUDA(ACC_GESVD_RELACC_CUSOLVER, GesvdRelAccSizes)

MINI_ACC_MAIN()

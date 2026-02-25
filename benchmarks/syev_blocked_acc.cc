#include <blas/functions.hh>
#include <blas/extra.hh>
#include <blas/linalg.hh>
#include <util/miniacc.hh>

#include "acc_utils.hh"
#include "miniacc_accuracy_common.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>

using namespace batchlas;

namespace {

template <typename Benchmark>
void EigensolverSizes(Benchmark* b) {
    for (double n : {16.0, 32.0, 64.0}) b->Args({n});
}

template <typename Real, Backend B>
void run_syev_blocked(miniacc::State& state) {
    const int n = std::max(2, state.arg_int(0));
    const double target_log10 = std::isfinite(state.target_log10_cond()) ? state.target_log10_cond() : 1.0;
    const int chunk_batch = miniacc_acc::chunk_batch_from_samples(state.samples());

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    state.SetTag("impl", "syev_blocked");
    state.SetTag("backend", miniacc_acc::backend_name<B>());
    state.SetTag("dtype", miniacc_acc::dtype_name<Real>());

    size_t produced = 0;
    while (produced < state.samples()) {
        const int cur_batch = static_cast<int>(std::min<size_t>(static_cast<size_t>(chunk_batch), state.samples() - produced));
        const unsigned seed = state.seed() + static_cast<unsigned>(produced);

        auto A = random_hermitian_with_log10_cond_metric<B, Real>(
            *q, n, static_cast<Real>(target_log10), NormType::Spectral, cur_batch, seed);

        std::vector<std::vector<double>> ref_eigs;
        std::vector<char> ref_ok;
        miniacc_acc::make_dense_reference(A, ref_eigs, ref_ok);

        auto A_work = A.clone();
        UnifiedVector<Real> eigvals(static_cast<size_t>(n) * static_cast<size_t>(cur_batch));

        try {
            UnifiedVector<std::byte> ws(
                syev_blocked_buffer_size<B, Real>(*q, A_work.view(), JobType::EigenVectors, Uplo::Lower, 32, 32));
            syev_blocked<B, Real>(*q,
                                  A_work.view(),
                                  eigvals.to_span(),
                                  JobType::EigenVectors,
                                  Uplo::Lower,
                                  ws.to_span(),
                                  32,
                                  32);
            q->wait();
        } catch (const std::exception& ex) {
            miniacc_acc::record_failed_samples(state, n, n, cur_batch, target_log10, std::string("solver_exception:") + ex.what());
            produced += static_cast<size_t>(cur_batch);
            continue;
        }

        VectorView<Real> evals_view(eigvals.to_span(), n, cur_batch, 1, n);
        auto conds = cond<B>(*q, A.view(), NormType::Spectral);
        q->wait();
        miniacc_acc::record_eigensolver_metrics<B, Real>(
            state,
            *q,
            n,
            false,
            A,
            evals_view,
            A_work,
            conds,
            ref_eigs,
            ref_ok);

        produced += static_cast<size_t>(cur_batch);
    }
}

} // namespace

template <typename Real, Backend B>
static void ACC_SYEV_BLOCKED(miniacc::State& state) {
    run_syev_blocked<Real, B>(state);
}

template <typename Benchmark>
void EigensolverSizesNetlib(Benchmark* b) {
    EigensolverSizes(b);
}

BATCHLAS_REGISTER_ACCURACY(ACC_SYEV_BLOCKED, EigensolverSizes)

MINI_ACC_MAIN()

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
    for (double n : {16.0, 32.0, 64.0}) {
        for (double neigs : {n / 4, n / 2}) {
            for (double extra_dirs : {0, 2}) {
                b->Args({n, neigs, extra_dirs});
            }
        }
    }
}

template <typename Real, Backend B>
void run_syevx(miniacc::State& state) {
    const int n = std::max(2, state.arg_int(0));
    const int neigs = std::max(1, state.arg_int(1));
    const int extra_directions = std::max(0, state.arg_int(2));
    const bool find_largest = false;
    const double target_log10 = std::isfinite(state.target_log10_cond()) ? state.target_log10_cond() : 1.0;
    const int chunk_batch = miniacc_acc::chunk_batch_from_samples(state.samples());

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    state.SetTag("impl", "syevx");
    state.SetTag("backend", miniacc_acc::backend_name<B>());
    state.SetTag("dtype", miniacc_acc::dtype_name<Real>());

    SyevxParams<Real> params{};
    params.algorithm = OrthoAlgorithm::ShiftChol3;
    params.iterations = 50;
    params.extra_directions = extra_directions;
    params.find_largest = find_largest;
    params.absolute_tolerance = static_cast<Real>(1e-6);
    params.relative_tolerance = static_cast<Real>(1e-6);

    size_t produced = 0;
    while (produced < state.samples()) {
        const int cur_batch = static_cast<int>(std::min<size_t>(static_cast<size_t>(chunk_batch), state.samples() - produced));
        const unsigned seed = state.seed() + static_cast<unsigned>(produced);

        try {
            auto A = Matrix<Real, MatrixFormat::CSR>::RandomSparseHermitian(
                n,
                0.1f,
                cur_batch,
                seed,
                static_cast<Real>(10.0));
            auto A_dense = A.template convert_to<MatrixFormat::Dense>();
            std::vector<std::vector<double>> ref_eigs;
            std::vector<char> ref_ok;
            miniacc_acc::make_dense_reference(A_dense, ref_eigs, ref_ok);

            auto A_work = A.clone();
            UnifiedVector<Real> eigvals(static_cast<size_t>(neigs) * static_cast<size_t>(cur_batch));
            auto V = Matrix<Real>::Zeros(n, neigs, cur_batch);

            try {
                UnifiedVector<std::byte> ws(
                    syevx_buffer_size<B>(*q,
                                         A_work.view(),
                                         eigvals.to_span(),
                                         static_cast<size_t>(neigs),
                                         JobType::EigenVectors,
                                         V.view(),
                                         params));
                syevx<B>(*q,
                         A_work.view(),
                         eigvals.to_span(),
                         static_cast<size_t>(neigs),
                         ws.to_span(),
                         JobType::EigenVectors,
                         V.view(),
                         params);
                q->wait_and_throw();
            } catch (const std::exception& ex) {
                miniacc_acc::record_failed_samples(state, n, neigs, cur_batch, target_log10, std::string("solver_exception:") + ex.what());
                produced += static_cast<size_t>(cur_batch);
                continue;
            }

            VectorView<Real> evals_view(eigvals.to_span(), neigs, cur_batch, 1, neigs);
            auto conds = cond<B>(*q, A_dense.view(), NormType::Spectral);
            q->wait_and_throw();
            miniacc_acc::record_eigensolver_metrics<B, Real>(
                state,
                *q,
                neigs,
                find_largest,
                A_dense,
                evals_view,
                V,
                conds,
                ref_eigs,
                ref_ok);
        } catch (const std::exception& ex) {
            miniacc_acc::record_failed_samples(state, n, neigs, cur_batch, target_log10,
                                               std::string("setup_or_post_exception:") + ex.what());
        }

        produced += static_cast<size_t>(cur_batch);
    }
}

} // namespace

template <typename Real, Backend B>
static void ACC_SYEVX(miniacc::State& state) {
    run_syevx<Real, B>(state);
}

template <typename Benchmark>
void EigensolverSizesNetlib(Benchmark* b) {
    EigensolverSizes(b);
}

BATCHLAS_REGISTER_ACCURACY(ACC_SYEVX, EigensolverSizes)

MINI_ACC_MAIN()

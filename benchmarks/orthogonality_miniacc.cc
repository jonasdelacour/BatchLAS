#include <blas/functions.hh>
#include <blas/extra.hh>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <batchlas/backend_config.h>
#include "../src/queue.hh"

#include <util/miniacc.hh>
#include "acc_utils.hh"
#include "miniacc_accuracy_common.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

using namespace batchlas;

namespace {

template <typename Benchmark>
void OrthogonalityMiniaccSizes(Benchmark* b) {
    for (double n : {16.0, 32.0, 64.0}) {
        b->Args({n, n});
        b->Args({4.0 * n, n});
    }
}

template <typename Benchmark>
void OrthogonalityMiniaccSizesNetlib(Benchmark* b) {
    OrthogonalityMiniaccSizes(b);
}

template <typename Real, Backend B, OrthoAlgorithm Algo>
void run_ortho_case(miniacc::State& state, const char* impl_name) {
    const int m = std::max(1, state.arg_int(0));
    const int n_arg = state.arg_int(1);
    const int n = n_arg > 0 ? std::max(1, n_arg) : m;
    const double target_log10 = std::isfinite(state.target_log10_cond()) ? state.target_log10_cond() : 1.0;
    const int chunk_batch = miniacc_acc::chunk_batch_from_samples(state.samples());

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    state.SetTag("impl", impl_name);
    state.SetTag("backend", miniacc_acc::backend_name<B>());
    state.SetTag("dtype", miniacc_acc::dtype_name<Real>());

    if (m < n) {
        for (size_t i = 0; i < state.samples(); ++i) {
            state.RecordSample(
                {
                    {"orthogonality", std::numeric_limits<double>::quiet_NaN()},
                    {"log10_cond", target_log10}
                },
                false,
                "invalid_shape_m_lt_n");
        }
        return;
    }

    size_t produced = 0;
    while (produced < state.samples()) {
        const int cur_batch = static_cast<int>(std::min<size_t>(static_cast<size_t>(chunk_batch), state.samples() - produced));
        const unsigned int seed = state.seed() + static_cast<unsigned int>(produced);

        Matrix<Real, MatrixFormat::Dense> dense_A = (m == n)
            ? random_hermitian_with_log10_cond_metric<B, Real>(
                  *q,
                  n,
                  static_cast<Real>(target_log10),
                  NormType::Spectral,
                  cur_batch,
                  seed)
            : Matrix<Real, MatrixFormat::Dense>::Random(m, n, false, cur_batch, seed);
        q->wait();

        auto Q = dense_A.clone();
        UnifiedVector<std::byte> ws(ortho_buffer_size<B, Real>(*q, Q.view(), Transpose::NoTrans, Algo));
        ortho<B, Real>(*q, Q.view(), Transpose::NoTrans, ws.to_span(), Algo);
        q->wait();

        const auto ortho_vals = miniacc_acc::orthogonality_residuals<B, Real>(*q, Q);
        const bool have_measured_cond = (m == n);
        UnifiedVector<Real> conds = have_measured_cond
            ? cond<B>(*q, dense_A.view(), NormType::Spectral)
            : UnifiedVector<Real>();
        q->wait();

        for (int b = 0; b < cur_batch; ++b) {
            const double orth = static_cast<double>(ortho_vals[static_cast<size_t>(b)]);
            double log10_cond = target_log10;
            if (have_measured_cond) {
                const double cond = static_cast<double>(conds[static_cast<size_t>(b)]);
                log10_cond = std::log10(std::max(cond, 1e-300));
            }
            const bool ok = std::isfinite(orth);
            state.RecordSample(
                {
                    {"orthogonality", orth},
                    {"log10_cond", log10_cond}
                },
                ok,
                ok ? "" : "non_finite_orthogonality");
        }
            produced += static_cast<size_t>(cur_batch);
    }
}

template <typename Real, Backend B>
void ACC_ORTHO_CHOL2(miniacc::State& state) {
    run_ortho_case<Real, B, OrthoAlgorithm::Chol2>(state, "ortho_chol2");
}

template <typename Real, Backend B>
void ACC_ORTHO_CHOLESKY(miniacc::State& state) {
    run_ortho_case<Real, B, OrthoAlgorithm::Cholesky>(state, "ortho_cholesky");
}

template <typename Real, Backend B>
void ACC_ORTHO_SHIFTCHOL3(miniacc::State& state) {
    run_ortho_case<Real, B, OrthoAlgorithm::ShiftChol3>(state, "ortho_shiftchol3");
}

template <typename Real, Backend B>
void ACC_ORTHO_HOUSEHOLDER(miniacc::State& state) {
    run_ortho_case<Real, B, OrthoAlgorithm::Householder>(state, "ortho_householder");
}

template <typename Real, Backend B>
void ACC_ORTHO_CGS2(miniacc::State& state) {
    run_ortho_case<Real, B, OrthoAlgorithm::CGS2>(state, "ortho_cgs2");
}

template <typename Real, Backend B>
void ACC_ORTHO_SVQB(miniacc::State& state) {
    run_ortho_case<Real, B, OrthoAlgorithm::SVQB>(state, "ortho_svqb");
}

template <typename Real, Backend B>
void ACC_ORTHO_SVQB2(miniacc::State& state) {
    run_ortho_case<Real, B, OrthoAlgorithm::SVQB2>(state, "ortho_svqb2");
}

} // namespace

BATCHLAS_REGISTER_ACCURACY(ACC_ORTHO_CHOL2, OrthogonalityMiniaccSizes);
BATCHLAS_REGISTER_ACCURACY(ACC_ORTHO_CHOLESKY, OrthogonalityMiniaccSizes);
BATCHLAS_REGISTER_ACCURACY(ACC_ORTHO_SHIFTCHOL3, OrthogonalityMiniaccSizes);
BATCHLAS_REGISTER_ACCURACY(ACC_ORTHO_HOUSEHOLDER, OrthogonalityMiniaccSizes);
BATCHLAS_REGISTER_ACCURACY(ACC_ORTHO_CGS2, OrthogonalityMiniaccSizes);
BATCHLAS_REGISTER_ACCURACY(ACC_ORTHO_SVQB, OrthogonalityMiniaccSizes);
BATCHLAS_REGISTER_ACCURACY(ACC_ORTHO_SVQB2, OrthogonalityMiniaccSizes);

MINI_ACC_MAIN();

// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/miniacc.hh>

#include <benchmarks/acc_utils.hh>
#include <benchmarks/miniacc_accuracy_common.hh>

#include <complex>
#include <utility>

using namespace batchlas;

namespace {

template <typename Benchmark>
void GesvdCtaAccSizes(Benchmark* b) {
    for (double n : {8.0, 16.0, 32.0}) b->Args({n});
}

template <typename Scalar, Backend B>
void run_gesvd_cta_acc(miniacc::State& state) {
    miniacc_acc::run_gesvd_acc<Scalar, B>(
        state,
        "gesvd_cta",
        [](auto&&... args) { return gesvd_cta_buffer_size<B, Scalar>(std::forward<decltype(args)>(args)...); },
        [](auto&&... args) { return gesvd_cta<B, Scalar>(std::forward<decltype(args)>(args)...); });
}

template <typename Scalar, Backend B>
void ACC_GESVD_CTA(miniacc::State& state) {
    run_gesvd_cta_acc<Scalar, B>(state);
}

}  // namespace

BATCHLAS_ACC_CUDA_ALL_TYPES(ACC_GESVD_CTA, GesvdCtaAccSizes)

MINI_ACC_MAIN();

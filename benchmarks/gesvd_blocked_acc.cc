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
void GesvdAccSizes(Benchmark* b) {
    for (double n : {16.0, 32.0, 64.0, 128.0}) b->Args({n});
}

template <typename Scalar, Backend B>
void run_gesvd_blocked_acc(miniacc::State& state) {
    miniacc_acc::run_gesvd_acc<Scalar, B>(
        state,
        "gesvd_blocked",
        [](auto&&... args) { return gesvd_blocked_buffer_size<B, Scalar>(std::forward<decltype(args)>(args)...); },
        [](auto&&... args) { return gesvd_blocked<B, Scalar>(std::forward<decltype(args)>(args)...); });
}

} // namespace

template <typename Real, Backend B>
static void ACC_GESVD_BLOCKED(miniacc::State& state) {
    run_gesvd_blocked_acc<Real, B>(state);
}

template <typename Benchmark>
void GesvdAccSizesNetlib(Benchmark* b) {
    GesvdAccSizes(b);
}

BATCHLAS_REGISTER_ACCURACY_ALL_TYPES(ACC_GESVD_BLOCKED, GesvdAccSizes)

MINI_ACC_MAIN()

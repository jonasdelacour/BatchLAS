#pragma once

#include <array>

namespace batchlas::blas::dispatch {

enum class Provider {
    Auto,
    Vendor,
    BatchLAS_CTA,
    BatchLAS_Blocked,
    BatchLAS_TwoStage,
    BatchLAS_Jacobi,
    Netlib,
};

struct DispatchPolicy {
    Provider forced = Provider::Auto;
    // BatchLAS_Jacobi sits AFTER the CTA/Blocked paths deliberately. For real
    // input CTA precedes it and accepts every shape it accepts, so it is
    // reachable in exactly two useful ways: forced via BATCHLAS_GESVD_PROVIDER,
    // and automatically for complex GENERAL input, where the CTA and Blocked
    // predicates return false and dispatch currently falls through to Vendor and
    // throws. Promoting it ahead of CTA is a separate, measured change.
    std::array<Provider, 6> order = {
        Provider::BatchLAS_CTA,
        Provider::BatchLAS_Blocked,
        Provider::BatchLAS_TwoStage,
        Provider::BatchLAS_Jacobi,
        Provider::Vendor,
        Provider::Netlib,
    };
    bool log = false;
    bool require_in_order = false;
};

} // namespace batchlas::blas::dispatch

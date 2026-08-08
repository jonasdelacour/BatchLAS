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
    // This is the SHARED fallback order, used by every op that does not name
    // itself in default_order_for_op (blas/dispatch/env.hh).
    //
    // BatchLAS_Jacobi sits after the CTA/Blocked paths here, which is harmless:
    // gesvd is the only op whose chooser matches BatchLAS_Jacobi at all, and
    // gesvd no longer uses this array -- it has its own order, with Jacobi
    // first, and env.hh carries the measurement that put it there.
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

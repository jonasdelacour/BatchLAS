#pragma once

#include <array>

namespace batchlas::blas::dispatch {

enum class Provider {
    Auto,
    Vendor,
    BatchLAS_CTA,
    BatchLAS_Blocked,
    BatchLAS_TwoStage,
    Netlib,
};

struct DispatchPolicy {
    Provider forced = Provider::Auto;
    std::array<Provider, 5> order = {
        Provider::BatchLAS_CTA,
        Provider::BatchLAS_Blocked,
        Provider::BatchLAS_TwoStage,
        Provider::Vendor,
        Provider::Netlib,
    };
    bool log = false;
    bool require_in_order = false;
};

} // namespace batchlas::blas::dispatch

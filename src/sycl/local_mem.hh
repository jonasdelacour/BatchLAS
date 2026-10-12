#pragma once
// The local memory one work-group may launch with: what every SLM capacity is computed from.
// DPC++ reports the CUDA opt-in maximum (99 KiB on sm_89/sm_120) and opts launches in itself.
// acpp reports 48 KiB and never opts in; with BATCHLAS_ACPP_SLM_OPTIN the interposer
// (acpp_slm_optin.cc) does, and the budget is the driver's opt-in maximum.
// evidence: docs/design/sycl-implementations.md#sycl-impl-slm-budget
#include "impl.hh"
#include "../util/resident_capacity.hh"

#include <batchlas/error.hh>

#include <algorithm>
#include <cstddef>
#include <string>

#if BATCHLAS_SYCL_IMPL_ACPP && BATCHLAS_ACPP_SLM_OPTIN
extern "C" int batchlas_acpp_slm_optin_bytes(int cuda_ordinal);
#endif

namespace batchlas::impl {

inline std::size_t local_mem_bytes(const sycl::device& d) {
    const std::size_t reported = d.get_info<sycl::info::device::local_mem_size>();
#if BATCHLAS_SYCL_IMPL_ACPP && BATCHLAS_ACPP_SLM_OPTIN
    if (is_cuda(backend_of(d))) {
        const int optin = batchlas_acpp_slm_optin_bytes(sycl::get_native<kCudaBackend>(d));
        if (optin > 0) return std::max(reported, static_cast<std::size_t>(optin));
    }
#endif
    return reported;
}

// resident::cta_fit_wg_multiplier, throwing before a launch that cannot fit one step.
inline int cta_wg_multiplier(const char* who, int requested, int probs_per_step,
                             std::size_t bytes_per_prob, std::size_t local_mem) {
    const int m = resident::cta_fit_wg_multiplier(requested, probs_per_step, bytes_per_prob, local_mem);
    if (m < 1) {
        throw batchlas::unsupported(std::string(who) + ": one work-group needs " +
                                    std::to_string(static_cast<std::size_t>(probs_per_step) * bytes_per_prob) +
                                    " B of local memory; the device launches " + std::to_string(local_mem));
    }
    return m;
}

}  // namespace batchlas::impl

#pragma once
// The local memory one work-group may launch with: what every SLM capacity is computed from.
// DPC++ reports the CUDA opt-in maximum (99 KiB on sm_89/sm_120) and opts launches in itself.
// acpp reports 48 KiB; BATCHLAS_ACPP_SLM_OPTIN's interposer opts in to the driver maximum.
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

// acpp Metal appends wg * 8 B of group-algorithm scratch to every launch and refuses one past
// the device maximum (metal_queue.cpp), so wg lanes may request local_mem_size - 8 * wg.
inline std::size_t local_mem_scratch_per_item(const sycl::device& d) {
    return is_metal(backend_of(d)) ? std::size_t{8} : std::size_t{0};
}

inline std::size_t local_mem_bytes(const sycl::device& d, std::size_t wg) {
    std::size_t reported = d.get_info<sycl::info::device::local_mem_size>();
#if BATCHLAS_SYCL_IMPL_ACPP && BATCHLAS_ACPP_SLM_OPTIN
    if (is_cuda(backend_of(d))) {
        const int optin = batchlas_acpp_slm_optin_bytes(sycl::get_native<kCudaBackend>(d));
        if (optin > 0) reported = std::max(reported, static_cast<std::size_t>(optin));
    }
#endif
    const std::size_t scratch = local_mem_scratch_per_item(d) * wg;
    return reported > scratch ? reported - scratch : 0;
}

// What ANY work-group may request; a caller that knows its width may ask (d, wg) for more.
inline std::size_t local_mem_bytes(const sycl::device& d) {
    if (local_mem_scratch_per_item(d) == 0) return local_mem_bytes(d, 0);
    return local_mem_bytes(d, d.get_info<sycl::info::device::max_work_group_size>());
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

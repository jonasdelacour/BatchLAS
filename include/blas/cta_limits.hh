#pragma once

// Size policy shared by every CTA (one-work-group-per-problem) routine.
//
// Historically the CTA eigensolver chain (sytrd_cta -> steqr -> ormqx_cta) was
// capped at n <= 32.  That cap was structural, not a memory limit: the kernels
// used SubGroupPartition<P>, a chunk of a 32-lane warp, so P > 32 was silently
// wrong.  With WorkGroupPartition<P> (src/extensions/sg_compat.hh) the
// collectives run on shared memory and a real work-group barrier, so the only
// remaining limit is how much shared memory one problem needs.
//
// Per-problem shared-memory footprint of the widest CTA kernel in the chain:
//   sytrd_cta : P*P (tile) + 2P (v,w) + P (collective scratch)
//   ormqx_cta : P*(P+1) (row-padded C tile) + P (v)
// so P*(P+1) + 2P elements bounds both.
//
// Env var: BATCHLAS_CTA_LARGE_N=0 restores the historical n <= 32 cap
// (A/B switch; default is enabled).

#include <cstddef>
#include <cstdlib>
#include <string>

namespace batchlas {

inline bool cta_large_n_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("BATCHLAS_CTA_LARGE_N");
        if (!e) return true;
        const std::string s(e);
        return !(s == "0" || s == "off" || s == "OFF" || s == "false" || s == "FALSE" || s == "no");
    }();
    return enabled;
}

// Elements of shared memory one problem of width P needs.
inline std::size_t cta_partition_elems(std::size_t P) {
    return P * (P + 1) + 2 * P;
}

// Largest supported partition width (and hence largest supported n) for a
// scalar of size `elem_size` on a device offering `local_mem_bytes` of
// work-group local memory.  Always >= 32, so the legacy sub-group path is
// never taken away.
inline int cta_max_partition(std::size_t elem_size, std::size_t local_mem_bytes) {
    if (!cta_large_n_enabled()) return 32;
    if (elem_size == 0) return 32;
    int best = 32;
    for (int P : {64, 128}) {
        const std::size_t bytes = cta_partition_elems(static_cast<std::size_t>(P)) * elem_size;
        if (bytes <= local_mem_bytes) {
            best = P;
        } else {
            break;
        }
    }
    return best;
}

} // namespace batchlas

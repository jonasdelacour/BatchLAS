#pragma once
// amdgcn backend (stub: forwards to the generic sub-group backend).

#include "backend_generic.hh"

namespace batchlas::sgp {

template <uint32_t P, bool Masked>
struct AmdgcnBackend : GenericBackend<P, Masked> {
    static constexpr const char* name = "amdgcn-stub";
    static constexpr bool masked_by_default = false;
};

} // namespace batchlas::sgp

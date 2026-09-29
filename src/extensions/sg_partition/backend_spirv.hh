#pragma once
// spirv backend (stub: forwards to the generic sub-group backend).

#include "backend_generic.hh"

namespace batchlas::sgp {

template <uint32_t P, bool Masked>
struct SpirvBackend : GenericBackend<P, Masked> {
    static constexpr const char* name = "spirv-stub";
    static constexpr bool masked_by_default = false;
};

} // namespace batchlas::sgp

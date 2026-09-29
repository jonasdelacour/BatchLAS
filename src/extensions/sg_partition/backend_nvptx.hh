#pragma once
// nvptx backend (stub: forwards to the generic sub-group backend).

#include "backend_generic.hh"

namespace batchlas::sgp {

template <uint32_t P, bool Masked>
struct NvptxBackend : GenericBackend<P, Masked> {
    static constexpr const char* name = "nvptx-stub";
    static constexpr bool masked_by_default = true;
};

} // namespace batchlas::sgp

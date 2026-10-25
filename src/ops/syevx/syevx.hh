// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/extensions.hh>

namespace batchlas {

// Never Auto; deterministic so solve and sizing agree. Internal (syevx.cc, syevx_lobpcg.cc).
SyevxPreconditioner syevx_select_preconditioner(SyevxPreconditioner requested, bool iluk_configured,
                                                bool find_largest);

}  // namespace batchlas

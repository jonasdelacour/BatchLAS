// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/extensions.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-span.hh>

#include <src/queue.hh>
#include <src/util/batch_mask.hh>

#include <cstdint>
#include <string>

namespace batchlas {

// Never Auto; deterministic so solve and sizing agree. Internal (syevx.cc, syevx_lobpcg.cc).
SyevxPreconditioner syevx_select_preconditioner(SyevxPreconditioner requested, bool iluk_configured,
                                                bool find_largest);

namespace detail {

// The iterative arms (LOBPCG, Filtered) on a masked call: refused before any write wherever a
// child they reach refuses the mask. Asked by the dispatcher before its m fill, by each arm's
// entry and by each arm's size query. evidence: docs/design/masked-batches.md#masked-batches-refusal-policy
template <Backend B, typename T>
void syevx_iterative_refuse_mask(const Queue& /*ctx*/, const char* who, Span<const std::uint8_t> mask,
                                 const SyevxParams<T>& params, bool lobpcg) {
    if (mask.empty()) return;
    const std::string w(who);
    if constexpr (B == Backend::NETLIB) {
        throw batchlas::unsupported(w + ": batch mask not supported on the host backend, whose gemm is the host BLAS");
    }
    if (!lobpcg) return;
    if (params.preconditioner != nullptr || params.build_preconditioner) {
        // mask: lift after iluk (W4c)
        throw batchlas::unsupported(w + ": batch mask not supported with an ILU(k) preconditioner");
    }
    if (params.instrumentation != nullptr) {  // its histories are written per item on the host
        throw batchlas::unsupported(w + ": batch mask not supported with instrumentation");
    }
}

}  // namespace detail
}  // namespace batchlas

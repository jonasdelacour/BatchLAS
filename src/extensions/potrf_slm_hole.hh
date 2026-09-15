#pragma once

// The 48 KB dynamic-local-memory launch hole, spelled once for both potrf leaves: a request
// inside this band is refused at enqueue. evidence: docs/perf/potrf.md#the-48-kb-launch-hole

#include <cstddef>

namespace batchlas::potrf_native {

constexpr std::size_t kPotrfHoleLo = 47104;
constexpr std::size_t kPotrfHoleHi = 49664;
constexpr std::size_t kPotrfHolePadTo = 49920;

constexpr std::size_t potrf_hole_padded(std::size_t bytes) {
    return (bytes > kPotrfHoleLo && bytes <= kPotrfHoleHi) ? kPotrfHolePadTo : bytes;
}

}  // namespace batchlas::potrf_native

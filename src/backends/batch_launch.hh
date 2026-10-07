#pragma once

#include "../linalg-impl.hh"

namespace batchlas::backend {

// Host loop over an unbatched vendor level-3 call. `<= 1` is deliberate: an empty batch still issues one launch.
// evidence: docs/design/runtime-internals.md#runtime-internals-vendor-tus-instantiate-only-vendor-symbols
template <typename F, typename View, typename... Views>
inline void for_each_batch_item(F&& f, const View& first, const Views&... rest) {
    if (first.batch_size() <= 1) {
        f(first, rest...);
        return;
    }
    for (int batch = 0; batch < first.batch_size(); ++batch) {
        f(first[batch], rest[batch]...);
    }
}

} // namespace batchlas::backend

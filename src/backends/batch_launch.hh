#pragma once

#include "../linalg-impl.hh"

namespace batchlas::backend {

// The host loop that every unbatched vendor Level-3 call is wrapped in.
//
// cuBLAS and rocBLAS spell symm/hemm/syrk/herk/syr2k/her2k/trmm only for a
// single matrix -- there is no strided- or pointer-batched form of any of them
// -- so a batch is a host loop issuing one launch per member. Ten call sites
// across the two backends wrote that loop out by hand, and all ten wrote the
// same single-item short circuit with it: a batch of one is launched against
// the caller's own view rather than against view[0], so the common case does
// not pay for a sub-view it cannot use.
//
// `<= 1` rather than `== 1` is carried over verbatim from those sites: an empty
// view still issues exactly one launch, whose m or n the vendor then sees as
// zero. Tightening it here would silently change what an empty batch does,
// which is a decision for the callers, not for this loop.
//
// The callable comes first because the trailing pack is what varies: syrk and
// herk pass two views, everything else passes three, and the loop is otherwise
// identical. It is invoked with the batch members in the order the views were
// given, and the member count is taken from the first view, exactly as the
// hand-written loops took it from A.
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

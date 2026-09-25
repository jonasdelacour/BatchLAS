#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include <batchlas/util/sycl-span.hh>

#include "../queue.hh"

// The per-item convergence-status span, and the three things every tier does
// with one.
//
// THE CONTRACT (stated once, here; the public declarations in
// blas/extensions.hh point at it):
//
//   * `info` is one int32 per batch item, in the CALLER's memory. It is USM, so
//     a kernel writes it in place -- there is never a copy back, and no tier
//     needs workspace for it. That is what lets an empty span cost nothing and
//     leaves every *_buffer_size() result unchanged.
//   * 0 means the item converged. A value > 0 is LAPACK-like: the number of
//     off-diagonal elements that failed to converge, or 1 where a tier tracks
//     only "did not converge" and not a count.
//   * An EMPTY span means "not requested". info_ptr() then yields nullptr and
//     every helper below is a no-op, so the whole mechanism disappears.
//   * IT IS AN ACCUMULATOR, NOT AN OUTPUT REGISTER. Producers only ever raise a
//     value (info_report uses fetch_max); the span is zeroed exactly once, by
//     the entry point the CALLER invoked, via info_clear.
//
// That last rule is the load-bearing one. Nesting clears is harmless -- syev ->
// syev_blocked -> stedc clears the same span three times -- because no producer
// runs between them. Two clears with a producer BETWEEN them is the bug the rule
// prevents, and it is why stedc's merge kernels accumulate with fetch_max rather
// than storing: one stedc call runs many merges over the same batch items, and
// the answer wanted is "did any of them fail", not "did the last one".
//
// It is also why stedc does NOT hand its `info` to its leaf steqr calls. Both
// stedc drivers would clear it again underneath themselves -- the recursive one
// solves the two halves of the SAME items with two separate leaf solves, and the
// level-synchronous one solves leaves*batch problems in a single steqr call whose
// batch axis is longer than `info`. See the note in stedc.cc.
namespace batchlas::detail {

// The writable pointer, or nullptr when status was not requested.
//
// A too-short span is ignored rather than rejected, matching detail::info_target
// in src/linalg-impl.hh: the deducing overloads reject short spans up front
// (detail::require_info_span in blas/options.hh) and this layer is the library's
// inner-loop spelling, which must not pay a throw path per call.
inline int32_t* info_ptr(Span<int32_t> info, int64_t batch) {
    if (batch <= 0) return nullptr;
    if (info.size() < static_cast<size_t>(batch)) return nullptr;
    return info.data();
}

// Zero the span, once, in the caller-facing entry point. See the contract above.
inline void info_clear(Queue& ctx, Span<int32_t> info, int64_t batch) {
    if (int32_t* out = info_ptr(info, batch)) {
        ctx->memset(out, 0, sizeof(int32_t) * static_cast<size_t>(batch));
    }
}

// Raise item `item`'s status from inside a kernel. Device code: `info` is the
// pointer info_ptr returned, captured by value.
//
// fetch_max, not a store: the same item is written by several leaves, several
// merges and several sweeps of one solve, and the answer wanted is "did ANY of
// them fail", not "did the last one".
inline void info_report(int32_t* info, int64_t item, int32_t status) {
    if (!info || status <= 0) return;
    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>
        slot(info[item]);
    slot.fetch_max(status);
}

// Write item `item`'s status from inside a kernel that is the SINGLE writer for
// that item -- one CTA kernel per problem, with no other producer sharing the
// span. Device code: `info` is the pointer info_ptr returned, captured by value.
//
// A store, so a converged item is set to 0 without a separate clear. That is the
// point: a clear plus a raising kernel is two submissions, and nothing guarantees
// the caller's queue is in order, so the pair can race where a single store
// cannot. Use info_report instead wherever several producers share an item.
inline void info_store(int32_t* info, int64_t item, int32_t status) {
    if (!info) return;
    info[item] = status;
}

// Copy a LEAF tier's own per-item flag array into the caller's span.
//
// A STORE, not info_report's raise, and that is deliberate: this is for a tier
// that owns the whole answer for every item (bdsqr, syevx_lobpcg,
// syevx_filtered), so it is the single writer and needs no separate clear to
// order against -- which matters because a clear plus a fold is two kernels, and
// the queue is not guaranteed to be in order. Do NOT use it where several
// producers share the span; that is what info_report exists for.
//
// `one_means_converged` is not decoration: syevx_lobpcg and syevx_filtered both
// write 1 for CONVERGED, the opposite of LAPACK, so copying either verbatim
// would report failure on every healthy item and success on every broken one --
// and a one-directional test would not catch it.
//
// The caller keeps `flags` alive until the kernel has run. A pool draw already
// outlives the call (it is the caller's workspace); a local UnifiedVector does
// not, and syevx_filtered waits for exactly that reason.
inline void info_from_flags(Queue& ctx,
                            Span<int32_t> info,
                            const int32_t* flags,
                            int64_t batch,
                            bool one_means_converged) {
    int32_t* out = info_ptr(info, batch);
    if (!out || !flags) return;
    const bool invert = one_means_converged;
    ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(batch)), [=](sycl::id<1> id) {
            const int32_t flag = flags[id[0]];
            out[id[0]] = invert ? (flag != 0 ? 0 : 1) : (flag > 0 ? flag : 0);
        });
    });
}

// Map a kernel-side problem index to the caller's batch item.
//
// stedc's level-synchronous driver solves `nodes_per_item * batch` independent
// sub-problems in one launch, so its kernel index runs over a finer axis than
// `info`. Passing the divisor down is what keeps `info` a plain Span of length
// `batch` at every level: the alternative -- a per-level span -- would make the
// caller's array length depend on an internal tree shape.
inline int64_t info_item(int64_t problem, int64_t nodes_per_item) {
    return nodes_per_item > 1 ? problem / nodes_per_item : problem;
}

}  // namespace batchlas::detail

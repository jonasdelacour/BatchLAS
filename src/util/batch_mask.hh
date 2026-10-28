// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <batchlas/error.hh>
#include <batchlas/util/sycl-span.hh>

// The batch mask of one call: every operand view may carry one (MatrixView::with_batch_mask), and an
// op honours the single span they share. Host-side only, and it never reads the mask bytes.
// evidence: docs/design/known-defects.md#defect-12-vendor-families-accepted-a-heterogeneous-batch
namespace batchlas::detail {

inline Span<const std::uint8_t> as_const_mask(Span<const std::uint8_t> m) { return m; }
inline Span<const std::uint8_t> as_const_mask(Span<std::uint8_t> m) { return {m.data(), m.size()}; }

// Null for an unmasked call: the pointer a kernel captures and tests with `!mask || mask[b]`.
inline const std::uint8_t* mask_ptr(Span<const std::uint8_t> m) { return m.empty() ? nullptr : m.data(); }

template <typename View>
inline int64_t mask_view_batch(const View& v) { return static_cast<int64_t>(v.batch_size()); }

template <typename View>
inline bool mask_view_is_broadcast(const View& v, int64_t call_batch) {
    if (mask_view_batch(v) != call_batch) return true;
    if constexpr (requires { v.stride(); }) {
        return call_batch > 1 && v.stride() == 0;
    } else {
        return false;
    }
}

template <typename View>
inline void merge_call_mask(const char* op, const View& v, int64_t call_batch, Span<const std::uint8_t>& out) {
    const Span<const std::uint8_t> m = as_const_mask(v.batch_mask());
    if (m.empty()) return;
    if (mask_view_is_broadcast(v, call_batch)) {
        throw batchlas::invalid_argument(std::string(op) + ": a broadcast operand (batch " +
                                         std::to_string(mask_view_batch(v)) + " of " + std::to_string(call_batch) +
                                         ") cannot carry a batch mask");
    }
    if (out.empty()) { out = m; return; }
    if (out.data() != m.data() || out.size() != m.size()) {
        throw batchlas::invalid_argument(std::string(op) + ": operands carry different batch masks");
    }
}

// True when any operand view carries a batch mask: the correctness term a family adds to can_run
// until its kernels honour the mask.
template <class... V>
bool any_batch_mask(const V&... v) { return (v.has_batch_mask() || ...); }

// The one span shared by the masked operands whose batch is the call batch (the largest operand
// batch); empty when no operand is masked. Unmasked operands adopt it. Compares pointer and length.
// @throws batchlas::invalid_argument for two different spans, or a masked broadcast operand
template <typename... Views>
Span<const std::uint8_t> call_mask(const char* op, const Views&... v) {
    int64_t call_batch = 0;
    ((call_batch = mask_view_batch(v) > call_batch ? mask_view_batch(v) : call_batch), ...);
    Span<const std::uint8_t> out;
    (merge_call_mask(op, v, call_batch, out), ...);
    return out;
}

}  // namespace batchlas::detail

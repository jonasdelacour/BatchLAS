// SPDX-License-Identifier: MIT
#pragma once
// The per-family batch-mask contract: one check every migration wave runs per op and pinned
// family. An inactive item's outputs are bit-for-bit untouched; an active item equals the unmasked
// whole-batch call on the same pin. The mask is device-only (sycl::malloc_device, written by a
// kernel), so a host read faults on a GPU. evidence: docs/design/masked-batches.md#masked-batches-tests
#include <batchlas/error.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <gtest/gtest.h>

#include <src/queue.hh>
#include <src/select/select.hh>
#include <tests/test_support/coverage_rows.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace masked_contract {

/// One per-item output: `items` items of `item_bytes` bytes, `stride_bytes` apart from `base`.
struct Output {
    std::string name;
    const std::byte* base = nullptr;
    std::size_t item_bytes = 0, stride_bytes = 0;

    /// A matrix view's whole stride per item (ld pads and the item gap included).
    template <typename View>
    static Output matrix(std::string name, const View& v) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(v.data_ptr())>>;
        const std::size_t s = static_cast<std::size_t>(v.stride()) * sizeof(T);
        return {std::move(name), reinterpret_cast<const std::byte*>(v.data_ptr()), s, s};
    }
    /// `per_item` elements per item, packed (w, tau, pivots: per_item = n; info: 1).
    template <typename T>
    static Output span(std::string name, std::span<const T> s, std::size_t per_item) {
        const std::size_t b = per_item * sizeof(T);
        return {std::move(name), reinterpret_cast<const std::byte*>(s.data()), b, b};
    }
};

/// A mask pattern; `bytes` empty is the unmasked call. `chunk_items` > 0 runs it (and its
/// reference) under BATCHLAS_TUNE_L2_CHUNK_ITEMS.
struct Pattern {
    std::string name;
    std::vector<std::uint8_t> bytes;
    int chunk_items = 0;
    bool active(int b) const { return bytes.empty() || bytes[static_cast<std::size_t>(b)] != 0; }
};

inline Pattern by_rule(std::string name, int batch, const std::function<bool(int)>& on, int chunk = 0) {
    Pattern p{std::move(name), std::vector<std::uint8_t>(static_cast<std::size_t>(batch)), chunk};
    for (int b = 0; b < batch; ++b) p.bytes[static_cast<std::size_t>(b)] = on(b) ? 1 : 0;
    return p;
}

/// none, all-active, alternating (item 0 off), single; with `wg_items` > 1 the packed work-group
/// holes; with `chunk_items` > 0 the part boundaries and one wholly inactive part.
inline std::vector<Pattern> patterns(int batch, int wg_items = 0, int chunk_items = 0) {
    std::vector<Pattern> out{{"none", {}, 0},
                             by_rule("all-active", batch, [](int) { return true; }),
                             by_rule("alternating", batch, [](int b) { return b % 2 == 1; }),
                             by_rule("single", batch, [&](int b) { return b == batch / 2; })};
    if (wg_items > 1) {
        const int w = wg_items;
        out.push_back(by_rule("wg-slot0-off", batch, [=](int b) { return b % w != 0; }));
        out.push_back(by_rule("wg-last-only", batch, [=](int b) { return b % w == w - 1; }));
        out.push_back(by_rule("wg-interior-off", batch, [=](int b) { return b % w == 0 || b % w == w - 1; }));
    }
    if (chunk_items > 0 && batch > chunk_items) {
        const int k = chunk_items;
        out.push_back(by_rule("chunk-edges-off", batch, [=](int b) { return b % k != 0 && b % k != k - 1; }, k));
        out.push_back(by_rule("part-inactive", batch, [=](int b) { return b / k != 1; }, k));
    }
    return out;
}

/// Device-only mask bytes written by a kernel from a host copy.
class DeviceMask {
public:
    DeviceMask(batchlas::Queue& q, const std::vector<std::uint8_t>& bytes) : q_(q), n_(bytes.size()) {
        if (n_ == 0) return;
        batchlas::UnifiedVector<std::uint8_t> host(n_);
        std::copy(bytes.begin(), bytes.end(), host.data());
        ptr_ = sycl::malloc_device<std::uint8_t>(n_, *q_.operator->());
        std::uint8_t* d = ptr_;
        const std::uint8_t* h = host.data();
        q_->parallel_for(sycl::range<1>(n_), [=](sycl::id<1> i) { d[i[0]] = h[i[0]]; });
        q_.wait();
    }
    ~DeviceMask() {
        if (ptr_) sycl::free(ptr_, *q_.operator->());
    }
    DeviceMask(const DeviceMask&) = delete;
    DeviceMask& operator=(const DeviceMask&) = delete;
    batchlas::Span<const std::uint8_t> span() const { return {ptr_, ptr_ ? n_ : 0}; }

private:
    batchlas::Queue& q_;
    std::size_t n_;
    std::uint8_t* ptr_ = nullptr;
};

/// What the op under test supplies. `call` attaches the mask to every operand view (empty: the
/// unmasked call) and runs on `ws`; `prepare` rewrites every input and poisons every output the
/// same way each time; `outputs` lists every per-item output (matrices, w, info, pivots, tau).
struct Case {
    int batch = 0;
    std::function<void()> prepare;
    std::function<void(batchlas::Span<const std::uint8_t> mask, batchlas::Span<std::byte> ws)> call;
    std::function<std::vector<Output>()> outputs;
    /// The workspace query on a view with this mask; null: the op takes none.
    std::function<std::size_t(batchlas::Span<const std::uint8_t> mask)> buffer_size;
    /// EXPECT_VERIFY mode (the documented cross-item exceptions): checks the active items itself
    /// instead of the bit-for-bit comparison with the unmasked call.
    std::function<void(const std::string& pattern, std::span<const int> active)> verify_active;
    /// Every item holds the same input (the saturating-batch case): active items bit-identical.
    bool identical_items = false;
    /// Composites and Auto on a vendor backend: an unmasked child may take the vendor, which a
    /// masked one refuses, so a masked pattern's active items are compared with the all-active
    /// masked call (routing keys on capacity, never on the mask's contents).
    bool masked_reference = false;
    int wg_items = 0, chunk_items = 0;  ///< see patterns()
};

namespace detail {
using Bytes = std::vector<std::vector<std::byte>>;

inline Bytes snapshot(const std::vector<Output>& outs, int batch) {
    Bytes s;
    for (const Output& o : outs) s.emplace_back(o.base, o.base + o.stride_bytes * static_cast<std::size_t>(batch));
    return s;
}

inline bool item_equal(const std::vector<std::byte>& a, const std::vector<std::byte>& b, const Output& o, int item,
                       int item_b = -1) {
    const std::size_t off = o.stride_bytes * static_cast<std::size_t>(item);
    const std::size_t off_b = o.stride_bytes * static_cast<std::size_t>(item_b < 0 ? item : item_b);
    return std::memcmp(a.data() + off, b.data() + off_b, o.item_bytes) == 0;
}

inline Bytes run(batchlas::Queue& q, const Case& c, const Pattern& p, std::size_t ws_bytes, Bytes* before) {
    std::optional<batchlas::ScopedEnvVar> chunk;
    if (p.chunk_items > 0) chunk.emplace("BATCHLAS_TUNE_L2_CHUNK_ITEMS", std::to_string(p.chunk_items).c_str());
    batchlas::UnifiedVector<std::byte> ws(std::max<std::size_t>(ws_bytes, 1));
    c.prepare();
    q.wait();
    const auto outs = c.outputs();
    if (before) *before = snapshot(outs, c.batch);
    DeviceMask mask(q, p.bytes);
    c.call(mask.span(), {ws.data(), ws_bytes});
    q.wait();
    return snapshot(outs, c.batch);
}
}  // namespace detail

/// Runs every pattern of `c` and checks the contract. With `op` set, it first pins `spelling`
/// (strict) and requires a coverage `reached` row of op on that spelling for a masked call.
template <typename Choice>
void check(batchlas::Queue& q, const char* op, const std::string& spelling, const Case& c) {
    using batchlas::select::ScopedPin;
    using batchlas::select::StrictPin;
    const auto pats = patterns(c.batch, c.wg_items, c.chunk_items);
    std::optional<ScopedPin<Choice>> pin;
    if (op) {
        pin.emplace(op, spelling, StrictPin{});
        const auto rows = test_support::reached_rows([&] {
            const ScopedPin<Choice> inner(op, spelling, StrictPin{});
            detail::run(q, c, pats[2], c.buffer_size ? c.buffer_size(DeviceMask(q, pats[2].bytes).span()) : 0, nullptr);
        });
        EXPECT_TRUE(std::any_of(rows.begin(), rows.end(), [&](const auto& r) { return r.op == op && r.spelling == spelling; }))
            << op << ": the masked call never reached " << spelling;
    }
    std::optional<std::size_t> size;
    std::map<int, detail::Bytes> refs;  // the unmasked call per chunk setting
    std::map<int, detail::Bytes> masked_refs;  // masked_reference: the all-active call per chunk setting
    for (const Pattern& p : pats) {
        SCOPED_TRACE(std::string(op ? op : "op") + " " + spelling + " pattern " + p.name);
        std::size_t need = 0;
        if (c.buffer_size) {
            DeviceMask m(q, p.bytes);
            need = c.buffer_size(m.span());
            if (!p.bytes.empty()) {
                if (size) EXPECT_EQ(need, *size) << "buffer_size depends on the mask's contents";
                size = need;
            }
        }
        if (!refs.count(p.chunk_items)) {
            const Pattern none{"none", {}, p.chunk_items};
            refs[p.chunk_items] = detail::run(q, c, none, c.buffer_size ? c.buffer_size({}) : 0, nullptr);
        }
        if (c.masked_reference && !p.bytes.empty() && !masked_refs.count(p.chunk_items)) {
            const Pattern all = by_rule("all-active", c.batch, [](int) { return true; }, p.chunk_items);
            DeviceMask m(q, all.bytes);
            masked_refs[p.chunk_items] = detail::run(q, c, all, c.buffer_size ? c.buffer_size(m.span()) : 0, nullptr);
        }
        const detail::Bytes& ref = c.masked_reference && !p.bytes.empty() ? masked_refs[p.chunk_items] : refs[p.chunk_items];
        detail::Bytes before;
        const detail::Bytes got = detail::run(q, c, p, need, &before);
        const auto outs = c.outputs();
        std::vector<int> active;
        for (int b = 0; b < c.batch; ++b)
            if (p.active(b)) active.push_back(b);
        for (std::size_t o = 0; o < outs.size(); ++o)
            for (int b = 0; b < c.batch; ++b) {
                if (!p.active(b))
                    EXPECT_TRUE(detail::item_equal(got[o], before[o], outs[o], b))
                        << outs[o].name << ": inactive item " << b << " was written";
                else if (!c.verify_active)
                    EXPECT_TRUE(detail::item_equal(got[o], ref[o], outs[o], b))
                        << outs[o].name << ": active item " << b << " differs from the unmasked call";
                if (c.identical_items && p.active(b) && !active.empty())
                    EXPECT_TRUE(detail::item_equal(got[o], got[o], outs[o], b, active.front()))
                        << outs[o].name << ": active item " << b << " differs from item " << active.front();
            }
        if (c.verify_active) c.verify_active(p.name, active);
    }
}

/// The contract without a pin, for a utility outside selection.
inline void check(batchlas::Queue& q, const Case& c) { check<std::monostate>(q, nullptr, "", c); }

}  // namespace masked_contract

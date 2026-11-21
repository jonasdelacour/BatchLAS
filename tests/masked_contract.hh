// SPDX-License-Identifier: MIT
#pragma once
// The per-family batch-mask contract: one check every migration wave runs per op and pinned
// family. An inactive item's outputs are bit-for-bit untouched; an active item equals the unmasked
// whole-batch call on the same pin. The mask is device-only (sycl::malloc_device, written by a
// kernel), so a host read faults on a GPU. evidence: docs/design/masked-batches.md#masked-batches-tests
#include <batchlas/blas/enums.hh>
#include <batchlas/error.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <gtest/gtest.h>

#include <src/queue.hh>
#include <src/select/select.hh>
#include <tests/test_support/coverage_rows.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace masked_contract {

/// One per-item output: `items` items of `item_bytes` bytes, `stride_bytes` apart from `base`.
struct Output {
    std::string name;
    const std::byte* base = nullptr;
    std::size_t item_bytes = 0, stride_bytes = 0;
    std::size_t real_bytes = 0;  ///< 4 / 8: float / double reals (complex too); 0: compared as bytes

    template <typename T>
    static constexpr std::size_t real_bytes_of() {
        if constexpr (std::is_floating_point_v<T>) return sizeof(T);
        else if constexpr (batchlas::is_std_complex_v<T>) return sizeof(T) / 2;
        else return 0;
    }
    /// A matrix view's whole stride per item (ld pads and the item gap included).
    template <typename View>
    static Output matrix(std::string name, const View& v) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(v.data_ptr())>>;
        const std::size_t s = static_cast<std::size_t>(v.stride()) * sizeof(T);
        return {std::move(name), reinterpret_cast<const std::byte*>(v.data_ptr()), s, s, real_bytes_of<T>()};
    }
    /// `per_item` elements per item, packed (w, tau, pivots: per_item = n; info: 1).
    template <typename T>
    static Output span(std::string name, std::span<const T> s, std::size_t per_item) {
        const std::size_t b = per_item * sizeof(T);
        return {std::move(name), reinterpret_cast<const std::byte*>(s.data()), b, b, real_bytes_of<T>()};
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

/// The patterns check() runs: patterns(), or for an identical_items (saturating) case the four
/// {none, all-active, alternating, wg-slot0-off}.
template <typename C>
std::vector<Pattern> patterns_for(const C& c) {
    auto out = patterns(c.batch, c.wg_items, c.chunk_items);
    if (c.identical_items)
        std::erase_if(out, [](const Pattern& p) {
            return p.name != "none" && p.name != "all-active" && p.name != "alternating" && p.name != "wg-slot0-off";
        });
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

/// A host-side copy, for the one-off refusal checks outside check().
inline Bytes snapshot(const std::vector<Output>& outs, int batch) {
    Bytes s;
    for (const Output& o : outs) s.emplace_back(o.base, o.base + o.stride_bytes * static_cast<std::size_t>(batch));
    return s;
}

// Device to host by q.memcpy (no host walk over shared pages); a plain memcpy for host memory.
inline Bytes snapshot(batchlas::Queue& q, const std::vector<Output>& outs, int batch) {
    Bytes s;
    for (const Output& o : outs) {
        auto& dst = s.emplace_back(o.stride_bytes * static_cast<std::size_t>(batch));
        if (dst.empty()) continue;
        if (sycl::get_pointer_type(o.base, q->get_context()) == sycl::usm::alloc::unknown)
            std::memcpy(dst.data(), o.base, dst.size());
        else
            q->memcpy(dst.data(), o.base, dst.size());
    }
    q.wait();
    return s;
}

inline bool item_equal(const std::vector<std::byte>& a, const std::vector<std::byte>& b, const Output& o, int item,
                       int item_b = -1) {
    const std::size_t off = o.stride_bytes * static_cast<std::size_t>(item);
    const std::size_t off_b = o.stride_bytes * static_cast<std::size_t>(item_b < 0 ? item : item_b);
    return std::memcmp(a.data() + off, b.data() + off_b, o.item_bytes) == 0;
}

// n reals at a and b agree to their last bits: each pair is bit-equal, or finite and apart by at
// most 2^-10 of the larger of the two and the rms of the pairs that differ.
template <typename R>
bool reals_near(const std::byte* a, const std::byte* b, std::size_t n) {
    double ss = 0;
    std::size_t differ = 0;
    const auto at = [](const std::byte* p, std::size_t i) {
        R v;
        std::memcpy(&v, p + i * sizeof(R), sizeof(R));
        return static_cast<double>(v);
    };
    for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(a + i * sizeof(R), b + i * sizeof(R), sizeof(R)) == 0) continue;
        const double x = at(a, i), y = at(b, i);
        if (!std::isfinite(x) || !std::isfinite(y)) return false;
        ss += std::max(x * x, y * y);
        ++differ;
    }
    const double rms = differ ? std::sqrt(ss / double(differ)) : 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = at(a, i), y = at(b, i);
        if (std::fabs(x - y) > 0x1p-10 * std::max({std::fabs(x), std::fabs(y), rms})) return false;
    }
    return true;
}
// item_equal for an untyped output; for a typed one, reals_near over the item.
inline bool item_near(const std::vector<std::byte>& a, const std::vector<std::byte>& b, const Output& o, int item) {
    const std::size_t off = o.stride_bytes * static_cast<std::size_t>(item);
    if (o.real_bytes == 4) return reals_near<float>(a.data() + off, b.data() + off, o.item_bytes / 4);
    if (o.real_bytes == 8) return reals_near<double>(a.data() + off, b.data() + off, o.item_bytes / 8);
    return item_equal(a, b, o, item);
}

/// One workspace per check, grown to the largest size asked and poisoned before every run.
class Workspace {
public:
    static constexpr int kPoison = 0xA5;
    explicit Workspace(batchlas::Queue& q) : q_(q) {}
    batchlas::Span<std::byte> poisoned(std::size_t bytes) {
        if (!buf_ || buf_->size() < bytes) buf_.emplace(std::max<std::size_t>(bytes, 1));
        q_->memset(buf_->data(), kPoison, buf_->size());
        return {buf_->data(), bytes};
    }

private:
    batchlas::Queue& q_;
    std::optional<batchlas::UnifiedVector<std::byte>> buf_;
};

inline Bytes run(batchlas::Queue& q, const Case& c, const Pattern& p, const DeviceMask& mask, std::size_t ws_bytes,
                 Workspace& ws, Bytes* before) {
    std::optional<batchlas::ScopedEnvVar> chunk;
    if (p.chunk_items > 0) chunk.emplace("BATCHLAS_TUNE_L2_CHUNK_ITEMS", std::to_string(p.chunk_items).c_str());
    c.prepare();
    const auto span = ws.poisoned(ws_bytes);
    q.wait();
    const auto outs = c.outputs();
    if (before) *before = snapshot(q, outs, c.batch);
    c.call(mask.span(), span);
    q.wait();
    return snapshot(q, outs, c.batch);
}
}  // namespace detail

/// Runs every pattern of `c` (patterns_for) once and checks the contract. Each pattern is compared
/// with its chunk setting's reference: the unmasked call (`none`, which is that reference at no
/// chunk), or under masked_reference the all-active masked call (the unmasked call then never
/// runs); verify_active runs no reference. With `op` set, it pins `spelling` (strict) and requires
/// a coverage `reached` row of op on that spelling, captured in-process around the alternating run.
template <typename Choice>
void check(batchlas::Queue& q, const char* op, const std::string& spelling, const Case& c) {
    using batchlas::select::ScopedPin;
    using batchlas::select::StrictPin;
    std::optional<ScopedPin<Choice>> pin;
    if (op) pin.emplace(op, spelling, StrictPin{});
    const bool bitwise = !c.verify_active;
    const bool masked_reference = c.masked_reference;
    // On an Apple GPU (Metal) bits agree only within one kernel build, and the masked and unmasked
    // calls' builds differ in their last bits: there an active item agrees with the unmasked call to
    // its last bits (detail::item_near) and with the all-active masked call bit for bit.
    const bool one_build = !masked_reference && bitwise && q.device().type == batchlas::DeviceType::GPU &&
                           q.device().get_vendor() == batchlas::Vendor::APPLE;
    std::map<int, detail::Bytes> all_active;  // one_build: per chunk setting
    const auto is_reference = [&](const Pattern& p) {
        return masked_reference ? p.name == "all-active" && p.chunk_items == 0 : p.bytes.empty();
    };
    detail::Workspace ws(q);
    std::optional<std::size_t> size;
    std::map<int, detail::Bytes> refs;  // per chunk setting
    for (const Pattern& p : patterns_for(c)) {
        if (p.bytes.empty() && masked_reference && bitwise) continue;
        SCOPED_TRACE(std::string(op ? op : "op") + " " + spelling + " pattern " + p.name);
        const DeviceMask mask(q, p.bytes);
        std::size_t need = 0;
        if (c.buffer_size) {
            need = c.buffer_size(mask.span());
            if (!p.bytes.empty()) {
                if (size) EXPECT_EQ(need, *size) << "buffer_size depends on the mask's contents";
                size = need;
            }
        }
        // The reference first, so the reach capture below sees masked calls only.
        const bool self_reference = bitwise && is_reference(p) && !refs.count(p.chunk_items);
        if (bitwise && !self_reference && !refs.count(p.chunk_items)) {
            const Pattern r = masked_reference ? by_rule("all-active", c.batch, [](int) { return true; }, p.chunk_items)
                                                 : Pattern{"none", {}, p.chunk_items};
            const DeviceMask rm(q, r.bytes);
            refs[p.chunk_items] = detail::run(q, c, r, rm, c.buffer_size ? c.buffer_size(rm.span()) : 0, ws, nullptr);
        }
        detail::Bytes before, got;
        if (op && p.name == "alternating") {
            const auto rows = test_support::reached_rows([&] { got = detail::run(q, c, p, mask, need, ws, &before); });
            EXPECT_TRUE(std::any_of(rows.begin(), rows.end(), [&](const auto& r) { return r.op == op && r.spelling == spelling; }))
                << op << ": the masked call never reached " << spelling;
        } else {
            got = detail::run(q, c, p, mask, need, ws, &before);
        }
        if (self_reference) refs[p.chunk_items] = std::move(got);
        const detail::Bytes& res = self_reference ? refs[p.chunk_items] : got;
        const detail::Bytes* ref = bitwise && !self_reference ? &refs[p.chunk_items] : nullptr;
        const auto mr = all_active.find(p.chunk_items);
        const detail::Bytes* masked_ref = one_build && mr != all_active.end() ? &mr->second : nullptr;
        if (one_build && p.name == "all-active") all_active[p.chunk_items] = res;
        const auto outs = c.outputs();
        std::vector<int> active;
        for (int b = 0; b < c.batch; ++b)
            if (p.active(b)) active.push_back(b);
        for (std::size_t o = 0; o < outs.size(); ++o)
            for (int b = 0; b < c.batch; ++b) {
                if (!p.active(b))
                    EXPECT_TRUE(detail::item_equal(res[o], before[o], outs[o], b))
                        << outs[o].name << ": inactive item " << b << " was written";
                else if (ref && one_build) {
                    EXPECT_TRUE(detail::item_near(res[o], (*ref)[o], outs[o], b))
                        << outs[o].name << ": active item " << b << " differs from the unmasked call past its last bits";
                    if (masked_ref)
                        EXPECT_TRUE(detail::item_equal(res[o], (*masked_ref)[o], outs[o], b))
                            << outs[o].name << ": active item " << b << " differs from the all-active masked call";
                } else if (ref) {
                    EXPECT_TRUE(detail::item_equal(res[o], (*ref)[o], outs[o], b))
                        << outs[o].name << ": active item " << b << " differs from the "
                        << (masked_reference ? "all-active masked call" : "unmasked call");
                }
                if (c.identical_items && p.active(b) && !active.empty())
                    EXPECT_TRUE(detail::item_equal(res[o], res[o], outs[o], b, active.front()))
                        << outs[o].name << ": active item " << b << " differs from item " << active.front();
            }
        if (c.verify_active) c.verify_active(p.name, active);
    }
}

/// The contract without a pin, for a utility outside selection.
inline void check(batchlas::Queue& q, const Case& c) { check<std::monostate>(q, nullptr, "", c); }

}  // namespace masked_contract

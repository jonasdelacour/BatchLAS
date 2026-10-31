// SPDX-License-Identifier: MIT
#pragma once
// Bulk moves between a candidates harness's UnifiedVector buffers and the host. A host loop over
// malloc_shared pages migrates them a fault at a time once a kernel has touched them: that was
// ~97% of the wall time of a batch-1024 case (gemm_candidates_tests, trmm_candidates_tests).
#include <batchlas/sycl_interop.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/error.hh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace test_utils::usm_bulk {

// The fixture's Queue between attach() and detach(), else null: copies then fall back to the host.
inline batchlas::Queue* copy_queue = nullptr;
// Two host scratch slots, kept for one test case so a large download does not fault in fresh
// pages per problem; plain host memory, so no SYCL object outlives the runtime in a static.
inline std::unique_ptr<std::byte[]> scratch[2];
inline std::size_t scratch_bytes[2] = {0, 0};
// Grow-only device blocks on the attached queue (pointer, bytes, in use): a released block serves
// the next request it fits, and detach() frees them all, so a test case pays no malloc per problem.
struct DeviceBlock {
    std::byte* ptr = nullptr;
    std::size_t bytes = 0;
    bool busy = false;
};
inline std::vector<DeviceBlock> device_blocks;

inline void attach(batchlas::Queue* q) { copy_queue = q; }
inline void detach() {
    if (copy_queue) {
        sycl::queue& q = batchlas::sycl_queue(*copy_queue);
        q.wait();
        for (const DeviceBlock& b : device_blocks) sycl::free(b.ptr, q);
    }
    device_blocks.clear();
    copy_queue = nullptr;
    for (int i = 0; i < 2; ++i) {
        scratch[i].reset();
        scratch_bytes[i] = 0;
    }
}

// A device block of at least `bytes` on the attached queue, returned to the pool when the last
// owner lets go; the smallest idle block that fits, else a fresh one.
inline std::shared_ptr<std::byte> device_block(std::size_t bytes) {
    if (!copy_queue) throw batchlas::internal_error("usm_bulk::device_block needs an attached queue");
    DeviceBlock* best = nullptr;
    for (DeviceBlock& b : device_blocks)
        if (!b.busy && b.bytes >= bytes && (!best || b.bytes < best->bytes)) best = &b;
    if (!best) {
        auto* p = sycl::malloc_device<std::byte>(std::max<std::size_t>(bytes, 256), batchlas::sycl_queue(*copy_queue));
        if (!p) throw batchlas::internal_error("usm_bulk::device_block: malloc_device of " + std::to_string(bytes) + " bytes");
        device_blocks.push_back({p, std::max<std::size_t>(bytes, 256), false});
        best = &device_blocks.back();
    }
    best->busy = true;
    return std::shared_ptr<std::byte>(best->ptr, [](std::byte* p) {
        for (DeviceBlock& b : device_blocks)
            if (b.ptr == p) b.busy = false;
    });
}

// Copies `bytes` where `usm` (dst or src) lies in a UnifiedVector: on the attached queue when it is
// shared USM in that queue's context, else std::memcpy on the host (a CPU fixture, no fixture).
inline void copy(void* dst, const void* src, std::size_t bytes, const void* usm) {
    if (bytes == 0) return;
    if (copy_queue) {
        sycl::queue& q = batchlas::sycl_queue(*copy_queue);
        if (sycl::get_pointer_type(usm, q.get_context()) == sycl::usm::alloc::shared) {
            q.memcpy(dst, src, bytes).wait();
            return;
        }
    }
    std::memcpy(dst, src, bytes);
}

// A host copy of the `n` elements at `usm`, valid until the slot's next download or detach().
template <typename T>
const T* download(const T* usm, std::size_t n, int slot) {
    const std::size_t bytes = n * sizeof(T);
    if (scratch_bytes[slot] < bytes) {
        scratch[slot] = std::make_unique_for_overwrite<std::byte[]>(bytes);
        scratch_bytes[slot] = bytes;
    }
    usm_bulk::copy(scratch[slot].get(), usm, bytes, usm);
    return reinterpret_cast<const T*>(scratch[slot].get());
}

// The first index at which two host arrays of `n` elements differ bit for bit.
template <typename T>
std::optional<std::size_t> first_diff(const T* x, const T* y, std::size_t n) {
    if (std::memcmp(x, y, n * sizeof(T)) == 0) return std::nullopt;
    std::size_t e = 0;
    while (std::memcmp(x + e, y + e, sizeof(T)) == 0) ++e;
    return e;
}

// A test buffer of `total` elements as its harness writes it: segment g holds `batch` items
// `stride` apart from `off`, item it a copy of item it % reps of `rep`; `fill` everywhere else
// (gaps between and inside items included). The harness draws only the `reps` representatives.
template <typename T>
struct Pristine {
    struct Segment {
        std::size_t off = 0, stride = 0;
        std::vector<T> rep;
    };
    // An m x n footprint at `off` with leading dimension `ld` in every item of its segment.
    struct Hole {
        std::size_t off = 0, ld = 0, rows = 0, cols = 0;
    };
    std::vector<Segment> segs;
    T fill{};
    std::size_t total = 0, batch = 0, reps = 0;

    Pristine() = default;
    // segs: (offset, stride), ascending, each item inside its stride, segments disjoint.
    Pristine(std::size_t total_, std::size_t batch_, std::size_t reps_, T fill_,
             std::initializer_list<std::pair<std::size_t, std::size_t>> segs_)
        : fill(fill_), total(total_), batch(batch_), reps(std::min(reps_, batch_)) {
        if (batch > 0 && reps == 0) throw batchlas::internal_error("a non-empty batch needs a representative");
        for (const auto& [off, stride] : segs_) segs.push_back({off, stride, std::vector<T>(stride * reps, fill)});
    }

    // The element at buffer index e of a representative item (it < reps).
    T& at(std::size_t e) {
        dev.reset();
        for (auto g = segs.rbegin(); g != segs.rend(); ++g)
            if (g->off <= e) return g->rep.at(e - g->off);
        throw batchlas::internal_error("no segment holds element " + std::to_string(e));
    }

    // Device copy of buffer index e of a representative item (it < reps): the segments' reps back
    // to back in one grow-only block, uploaded by the first write() on an attached queue. A view
    // at rep(off) with the segment's stride holds the reps items of the original buffer.
    T* rep(std::size_t e) const {
        if (!dev) throw batchlas::internal_error("Pristine::rep before a write() on an attached queue");
        std::size_t base = 0;
        for (std::size_t g = 0; g < segs.size(); ++g) {
            if (g + 1 == segs.size() || segs[g + 1].off > e) return reinterpret_cast<T*>(dev.get()) + base + (e - segs[g].off);
            base += segs[g].rep.size();
        }
        throw batchlas::internal_error("no segment holds element " + std::to_string(e));
    }

    // Writes the buffer into `usm` (a UnifiedVector of `total`): the representatives, then each
    // segment's prefix doubled in place, a multiple of reps items at every step. On an attached
    // queue all of it is device copies from the rep() image (uploaded once) and one wait.
    void write(T* usm) const {
        if (copy_queue) return write_on_device(usm);
        auto fill_run = [&](std::size_t from, std::size_t to) {
            if (to <= from) return;
            const std::vector<T> run(to - from, fill);
            usm_bulk::copy(usm + from, run.data(), run.size() * sizeof(T), usm);
        };
        std::size_t cursor = 0;
        for (const Segment& g : segs) {
            fill_run(cursor, g.off);
            usm_bulk::copy(usm + g.off, g.rep.data(), g.rep.size() * sizeof(T), usm);
            for (std::size_t have = reps; have < batch;) {
                const std::size_t more = std::min(have, batch - have);
                usm_bulk::copy(usm + g.off + have * g.stride, usm + g.off, more * g.stride * sizeof(T), usm);
                have += more;
            }
            cursor = g.off + batch * g.stride;
        }
        fill_run(cursor, total);
    }

    // The lowest index at which the host copy `got` (n elements; another size is a mismatch at
    // the shorter end) is not bit-identical to this buffer, skipping `hole` in every item of the
    // segment that holds it.
    std::optional<std::size_t> first_change(const T* got, std::size_t n, std::optional<Hole> hole) const {
        if (n != total) return std::min(n, total);
        std::optional<std::size_t> hit;
        auto cmp = [&](std::size_t e, const T* want, std::size_t len) {
            if (hit) return;
            if (auto d = usm_bulk::first_diff(got + e, want, len)) hit = e + *d;
        };
        auto cmp_fill = [&](std::size_t from, std::size_t to) {
            for (std::size_t e = from; e < to && !hit; ++e)
                if (std::memcmp(got + e, &fill, sizeof(T)) != 0) hit = e;
        };
        std::size_t cursor = 0;
        for (const Segment& g : segs) {
            cmp_fill(cursor, g.off);
            const bool holed = hole && hole->off >= g.off && hole->off < g.off + g.stride;
            for (std::size_t it = 0; it < batch && !hit; ++it) {
                const std::size_t base = g.off + it * g.stride;
                const T* want = g.rep.data() + (it % reps) * g.stride;
                std::size_t done = 0;
                for (std::size_t j = 0; holed && j < hole->cols; ++j) {
                    const std::size_t col = hole->off - g.off + j * hole->ld;
                    cmp(base + done, want + done, col - done);
                    done = col + hole->rows;
                }
                cmp(base + done, want + done, g.stride - done);
            }
            cursor = g.off + batch * g.stride;
        }
        cmp_fill(cursor, total);
        return hit;
    }

    // first_change() of `usm` (a UnifiedVector of `total` on the attached queue) as one kernel
    // against rep(), with no download. Inside `hole` an item it >= reps must instead equal item
    // it % reps of `usm` itself bit for bit; the lowest such index is the second result.
    std::pair<std::optional<std::size_t>, std::optional<std::size_t>> first_change_on_device(
        const T* usm, std::optional<Hole> hole) const {
        static_assert(sizeof(T) % sizeof(std::uint32_t) == 0);
        constexpr std::size_t W = sizeof(T) / sizeof(std::uint32_t);
        constexpr std::size_t kMaxSegs = 4;
        constexpr unsigned long long kNone = ~0ull;
        if (segs.size() > kMaxSegs) throw batchlas::internal_error("Pristine: more than 4 segments");
        sycl::queue& q = batchlas::sycl_queue(*copy_queue);
        if (!dev) write_on_device(nullptr);
        struct Seg {
            std::size_t off, stride, rep;
        };
        std::array<Seg, kMaxSegs> sg{};
        std::size_t base = 0;
        for (std::size_t g = 0; g < segs.size(); ++g) {
            sg[g] = {segs[g].off, segs[g].stride, base};
            base += segs[g].rep.size();
        }
        // The hole's segment (nsegs: none) and its offset inside that segment's item.
        std::size_t hseg = kMaxSegs, hrel = 0, hld = 1, hrows = 0, hcols = 0;
        for (std::size_t g = 0; hole && g < segs.size(); ++g)
            if (hole->off >= segs[g].off && hole->off < segs[g].off + segs[g].stride)
                hseg = g, hrel = hole->off - segs[g].off, hld = hole->ld, hrows = hole->rows, hcols = hole->cols;
        // The two hit slots, then the fill's words: every ref below is a global pointer (a ref that
        // may be a captured array fails Metal's JIT, "conflicting address space in PHI node").
        constexpr std::size_t kHit = 2 * sizeof(unsigned long long);
        std::array<std::byte, kHit + sizeof(T)> init;
        const unsigned long long none[2] = {kNone, kNone};
        std::memcpy(init.data(), none, kHit);
        std::memcpy(init.data() + kHit, &fill, sizeof(T));
        auto out = device_block(init.size());
        auto* hit = reinterpret_cast<unsigned long long*>(out.get());
        const auto* fw = reinterpret_cast<const std::uint32_t*>(out.get() + kHit);
        q.memcpy(out.get(), init.data(), init.size()).wait();
        const auto* got = reinterpret_cast<const std::uint32_t*>(usm);
        const auto* want = reinterpret_cast<const std::uint32_t*>(dev.get());
        const std::size_t nsegs = segs.size(), nbatch = batch, nreps = reps;
        q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
             const std::size_t e = id[0];
             std::size_t g = kMaxSegs;
             for (std::size_t k = 0; k < nsegs; ++k)
                 if (e >= sg[k].off) g = k;
             const std::uint32_t* ref = fw;
             bool in_hole_rep = false;
             if (g < kMaxSegs && e < sg[g].off + nbatch * sg[g].stride) {
                 const std::size_t rel = e - sg[g].off, it = rel / sg[g].stride, r = rel % sg[g].stride;
                 const bool holed = g == hseg && r >= hrel && (r - hrel) / hld < hcols && (r - hrel) % hld < hrows;
                 if (holed && it < nreps) return;
                 if (holed) {
                     ref = got + (sg[g].off + (it % nreps) * sg[g].stride + r) * W;
                     in_hole_rep = true;
                 } else {
                     ref = want + (sg[g].rep + (it % nreps) * sg[g].stride + r) * W;
                 }
             }
             for (std::size_t w = 0; w < W; ++w)
                 if (got[e * W + w] != ref[w]) {
                     sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>(hit[in_hole_rep ? 1 : 0])
                         .fetch_min(static_cast<unsigned long long>(e));
                     return;
                 }
         }).wait();
        unsigned long long res[2];
        q.memcpy(res, hit, sizeof res).wait();
        auto opt = [](unsigned long long v) { return v == kNone ? std::nullopt : std::optional<std::size_t>(v); };
        return {opt(res[0]), opt(res[1])};
    }

private:
    mutable std::shared_ptr<std::byte> dev;  // rep(): the reps of every segment, back to back

    // write() on the attached queue; usm == nullptr only uploads the rep() image.
    void write_on_device(T* usm) const {
        sycl::queue& q = batchlas::sycl_queue(*copy_queue);
        const bool ordered = q.is_in_order();
        auto step = [&](sycl::event ev) {
            if (!ordered) ev.wait();
        };
        if (!dev) {
            std::size_t n = 0;
            for (const Segment& g : segs) n += g.rep.size();
            dev = device_block(std::max<std::size_t>(n, 1) * sizeof(T));
            std::size_t base = 0;
            for (const Segment& g : segs) {
                q.memcpy(reinterpret_cast<T*>(dev.get()) + base, g.rep.data(), g.rep.size() * sizeof(T)).wait();
                base += g.rep.size();
            }
        }
        if (!usm) return;
        auto fill_run = [&](std::size_t from, std::size_t to) {
            if (to > from) step(q.fill(usm + from, fill, to - from));
        };
        std::size_t cursor = 0, base = 0;
        for (const Segment& g : segs) {
            fill_run(cursor, g.off);
            step(q.memcpy(usm + g.off, reinterpret_cast<const T*>(dev.get()) + base, g.rep.size() * sizeof(T)));
            for (std::size_t have = reps; have < batch;) {
                const std::size_t more = std::min(have, batch - have);
                step(q.memcpy(usm + g.off + have * g.stride, usm + g.off, more * g.stride * sizeof(T)));
                have += more;
            }
            cursor = g.off + batch * g.stride;
            base += g.rep.size();
        }
        fill_run(cursor, total);
        q.wait();
    }
};

}  // namespace test_utils::usm_bulk

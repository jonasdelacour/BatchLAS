#pragma once
// Bulk moves between a candidates harness's UnifiedVector buffers and the host. A host loop over
// malloc_shared pages migrates them a fault at a time once a kernel has touched them: that was
// ~97% of the wall time of a batch-1024 case (gemm_candidates_tests, trmm_candidates_tests).
#include <batchlas/sycl_interop.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <algorithm>
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

inline void attach(batchlas::Queue* q) { copy_queue = q; }
inline void detach() {
    copy_queue = nullptr;
    for (int i = 0; i < 2; ++i) {
        scratch[i].reset();
        scratch_bytes[i] = 0;
    }
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
        if (batch > 0 && reps == 0) throw std::logic_error("a non-empty batch needs a representative");
        for (const auto& [off, stride] : segs_) segs.push_back({off, stride, std::vector<T>(stride * reps, fill)});
    }

    // The element at buffer index e of a representative item (it < reps).
    T& at(std::size_t e) {
        for (auto g = segs.rbegin(); g != segs.rend(); ++g)
            if (g->off <= e) return g->rep.at(e - g->off);
        throw std::logic_error("no segment holds element " + std::to_string(e));
    }

    // Writes the buffer into `usm` (a UnifiedVector of `total`): the representatives from the
    // host, then each segment's prefix doubled in place, a multiple of reps items at every step.
    void write(T* usm) const {
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
};

}  // namespace test_utils::usm_bulk

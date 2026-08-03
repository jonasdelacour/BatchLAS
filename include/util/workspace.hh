#pragma once
#include <cstddef>

// Deliberately does not include util/sycl-span.hh: that header includes
// util/sycl-device-queue.hh, which includes this one. Span is only needed as a
// return type here, so a forward declaration is enough and the accessors that
// mention it are defined out of line.
template <typename T>
struct Span;

struct Queue;

namespace batchlas {

// A borrow of scratch memory from a Queue's workspace arena, released when the
// handle goes out of scope.
//
//     auto ws = ctx.workspace(foo_buffer_size<B>(ctx, A));
//     foo<B>(ctx, A, ws.span());
//
// Why this rather than a local UnifiedVector, which is what the convenience
// overloads used to do:
//
//   - A UnifiedVector frees its memory in its destructor, i.e. when the calling
//     function returns. The kernels using it have only been *enqueued* by then.
//     Every such call site was either relying on an explicit .wait() or was
//     quietly freeing memory out from under work still in flight (inv's
//     Matrix-returning overload was doing exactly that). Releasing a lease frees
//     nothing -- the memory belongs to the Queue -- so the pointer stays valid.
//   - A fresh USM allocation per call is expensive, and these are per-call
//     scratch buffers whose sizes repeat.
//
// Released bytes are handed to the *next* lease, though, so the in-flight
// question does not disappear entirely -- it changes from "freed under running
// kernels" to "overwritten by later ones". On an in-order queue that is safe by
// construction, since the later work is ordered behind the earlier work. On an
// out-of-order queue, wait before letting the lease go.
//
// Leases nest, and the arena never moves memory that is currently leased: a
// borrow that does not fit in the current block opens a new one rather than
// reallocating, so an outer lease's pointer stays valid while an inner lease is
// live. Release order must still be the reverse of acquisition order, which
// scope-bound handles give you for free.
//
// A lease is tied to one Queue and is not thread-safe, in keeping with Queue
// itself.
class WorkspaceLease {
public:
    WorkspaceLease() = default;
    WorkspaceLease(const WorkspaceLease&) = delete;
    WorkspaceLease& operator=(const WorkspaceLease&) = delete;

    WorkspaceLease(WorkspaceLease&& other) noexcept
        : queue_(other.queue_), ptr_(other.ptr_), size_(other.size_),
          block_(other.block_), offset_(other.offset_) {
        other.queue_ = nullptr;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    WorkspaceLease& operator=(WorkspaceLease&& other) noexcept {
        if (this != &other) {
            release();
            queue_ = other.queue_;
            ptr_ = other.ptr_;
            size_ = other.size_;
            block_ = other.block_;
            offset_ = other.offset_;
            other.queue_ = nullptr;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~WorkspaceLease() { release(); }

    Span<std::byte> span() const;
    operator Span<std::byte>() const;

    std::byte* data() const { return ptr_; }
    std::size_t size() const { return size_; }

    // Give the bytes back before the handle goes out of scope. Idempotent.
    void release();

private:
    friend struct ::Queue;
    WorkspaceLease(Queue* q, std::byte* p, std::size_t n, std::size_t block, std::size_t offset)
        : queue_(q), ptr_(p), size_(n), block_(block), offset_(offset) {}

    Queue* queue_ = nullptr;
    std::byte* ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t block_ = 0;
    std::size_t offset_ = 0;
};

}  // namespace batchlas

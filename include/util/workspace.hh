#pragma once
#include <cstddef>
#include <cstdint>

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
// out-of-order queue nothing orders them, so release() waits on the queue before
// giving the bytes back; see the note on release() for what that costs.
//
// Leases nest, and the arena never moves memory that is currently leased: a
// borrow that does not fit in the current block opens a new one rather than
// reallocating, so an outer lease's pointer stays valid while an inner lease is
// live. Releasing in reverse order of acquisition -- which scope-bound handles
// give you for free -- is the case the arena is built for: it returns the bytes
// immediately. Releasing out of order is safe but wasteful: the arena will not
// re-serve memory underneath a live lease, so those bytes stay reserved until
// the leases taken after them are released too. Debug builds assert on it so the
// waste is discoverable rather than silent.
//
// Reassigning a live lease is that wasteful case, and unavoidably so:
//
//     ws = ctx.workspace(bigger);   // RHS is acquired before ws is released
//
// The new loan is taken on top of the old one, so the old one's bytes cannot be
// reclaimed until the new lease dies. In a loop that reassigns every iteration
// the arena therefore ratchets: iteration k stacks a k-th loan rather than
// reusing the same bytes, and the peak is only given back when the last lease
// goes. Call release() explicitly first if that matters:
//
//     ws.release();
//     ws = ctx.workspace(bigger);   // now the old bytes are the ones served
//
// This path does not assert -- it is a legal thing to write, and the arena
// cannot do better than it does -- so the ratchet is documented here rather than
// reported at runtime.
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
          block_(other.block_), offset_(other.offset_), seq_(other.seq_) {
        other.queue_ = nullptr;
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.seq_ = 0;
    }

    WorkspaceLease& operator=(WorkspaceLease&& other) noexcept {
        if (this != &other) {
            // Out-of-order by construction when both leases are on the same
            // queue: `other` was acquired before this line runs. See the note on
            // reassignment above; the arena is told not to assert on it.
            release_(/*diagnose_out_of_order=*/false);
            queue_ = other.queue_;
            ptr_ = other.ptr_;
            size_ = other.size_;
            block_ = other.block_;
            offset_ = other.offset_;
            seq_ = other.seq_;
            other.queue_ = nullptr;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.seq_ = 0;
        }
        return *this;
    }

    ~WorkspaceLease() { release(); }

    // Every accessor that hands out the borrowed memory is lvalue-only, and the
    // rvalue overload is deleted rather than absent. A lease is a scope guard:
    // the bytes go back to the arena when it dies, and the arena re-serves them
    // to the next borrow, so a pointer taken from a temporary lease is already
    // stale on the next line. All three of these compiled clean before, under
    // -Wall -Wextra -Wdangling, and all three aliased:
    //
    //     Span<std::byte> ws = ctx.workspace(n);      // lease dies here
    //     auto s = ctx.workspace(n).span();           // and here
    //     std::byte* p = ctx.workspace(n).data();     // and here
    //
    // The correct spelling differs from the first only by `auto`, which is why
    // the compiler has to be the one to say it. A const lvalue reference does
    // not do the job -- it binds to a prvalue -- so the accessors are
    // ref-qualified.
    Span<std::byte> span() const &;
    Span<std::byte> span() const && = delete;
    operator Span<std::byte>() const &;
    operator Span<std::byte>() const && = delete;

    std::byte* data() const & { return ptr_; }
    std::byte* data() const && = delete;
    std::size_t size() const { return size_; }

    // Give the bytes back before the handle goes out of scope. Idempotent, and
    // safe to call from a destructor: it never throws.
    //
    // Free on an in-order queue. On an out-of-order queue it blocks until the
    // queue is idle *when the release actually hands bytes back*, i.e. when this
    // is the innermost live lease, because the next borrow would otherwise be
    // written by a kernel the runtime is free to schedule against work still
    // reading these bytes. An out-of-order return costs nothing here: it only
    // marks the loan, and the later release that finally reclaims it does the
    // draining then. That stall is the price of the configuration -- a caller
    // that wants an out-of-order queue and no stall has to keep the lease alive
    // until it has waited on the work itself.
    //
    // Two things this drain does NOT cover, both worth knowing before relying on
    // it:
    //
    //   - It waits on the queue the lease was taken from, and nothing else. The
    //     dispatchers that need ordering (syev, gesvd, ormqr, iluk) build a
    //     *derived* in-order Queue from an out-of-order ctx and submit the
    //     kernels there, while the lease -- taken by the convenience overload
    //     above them -- belongs to ctx. Waiting on ctx says nothing about work
    //     submitted to the derived queue; what makes that safe is the derived
    //     queue being destroyed, and so drained, before the dispatcher returns.
    //     In general: a lease released on queue X does not order against work
    //     submitted to a queue derived from X.
    //   - Scope-bound leases inside the option-struct convenience overloads
    //     (blas/options.hh) are innermost, so on an out-of-order Queue those
    //     calls now block at scope exit where they used to return as soon as the
    //     work was enqueued.
    void release() noexcept;

private:
    friend struct ::Queue;
    // `diagnose_out_of_order` is forwarded to the arena; false suppresses the
    // debug assert for the one caller that cannot avoid returning out of order.
    void release_(bool diagnose_out_of_order) noexcept;

    WorkspaceLease(Queue* q, std::byte* p, std::size_t n, std::size_t block, std::size_t offset,
                   std::uint64_t seq)
        : queue_(q), ptr_(p), size_(n), block_(block), offset_(offset), seq_(seq) {}

    Queue* queue_ = nullptr;
    std::byte* ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t block_ = 0;
    std::size_t offset_ = 0;
    // Identifies this loan to the arena, which uses it to tell an innermost
    // release (rewind the cursor) from an out-of-order one (do not).
    std::uint64_t seq_ = 0;
};

}  // namespace batchlas

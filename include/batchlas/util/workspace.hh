#pragma once
#include <cstddef>
#include <cstdint>

#include <batchlas/export.hh>

/// @file
/// @brief WorkspaceLease: a scoped borrow of scratch memory from a Queue's workspace arena.
/// @see @ref design_workspace
/// @ingroup workspace

// No util/sycl-span.hh (include cycle through sycl-device-queue.hh): Span and Queue are forward
// declared, and these MUST stay inside namespace batchlas or they declare different types.
// evidence: docs/design/workspace.md#workspace-forward-declarations-in-workspacehh
namespace batchlas {

template <typename T>
struct Span;

struct Queue;

/// @brief A borrow of scratch memory from a Queue's workspace arena, returned to the arena when
/// the handle goes out of scope.
///
/// @code
/// auto ws = ctx.workspace(foo_buffer_size<B>(ctx, A));
/// foo<B>(ctx, A, ws.span());
/// @endcode
///
/// Releasing a lease frees nothing (the memory belongs to the Queue), so the pointer stays valid
/// for kernels still in flight; the bytes are handed to the *next* lease instead. On an in-order
/// Queue that is safe by construction. On an out-of-order Queue, release() drains the queue first
/// (see release()).
///
/// Leases nest, and the arena never moves leased memory. Release in reverse order of acquisition
/// (scope-bound handles do this for free); an out-of-order release is safe but keeps its bytes
/// reserved until every later lease is released, and debug builds assert on it.
/// @warning Reassigning a live lease (`ws = ctx.workspace(bigger);`) acquires the new loan before
///          the old one is returned, so a loop that reassigns every iteration ratchets the arena
///          upward. Call `ws.release()` first.
/// @note Tied to one Queue and not thread-safe, like Queue itself.
/// @see @ref design_workspace
/// @ingroup workspace
// Class-level BATCHLAS_API, not per member: it must cover the PRIVATE release_(), which the inline
// move-assignment calls. evidence: docs/design/symbol-visibility.md#symbol-visibility-workspacelease-is-exported-at-class-level
class BATCHLAS_API WorkspaceLease {
public:
    /// @brief An empty lease that owns nothing.
    WorkspaceLease() = default;
    WorkspaceLease(const WorkspaceLease&) = delete;
    WorkspaceLease& operator=(const WorkspaceLease&) = delete;

    /// @brief Take over `other`'s loan; `other` becomes empty.
    WorkspaceLease(WorkspaceLease&& other) noexcept
        : queue_(other.queue_), ptr_(other.ptr_), size_(other.size_),
          block_(other.block_), offset_(other.offset_), seq_(other.seq_) {
        other.queue_ = nullptr;
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.seq_ = 0;
    }

    /// @brief Release this lease's loan, then take over `other`'s.
    WorkspaceLease& operator=(WorkspaceLease&& other) noexcept {
        if (this != &other) {
            // Out-of-order by construction (`other` was acquired first): legal, so no assert.
            // evidence: docs/design/workspace.md#workspace-the-reassignment-ratchet
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

    /// @brief Returns the loan to the arena (see release()).
    ~WorkspaceLease() { release(); }

    /// @brief The borrowed bytes. Lvalue-only: the rvalue overload is deleted, because a span
    /// taken from a temporary lease is stale on the next line.
    // All three of `Span<std::byte> ws = ctx.workspace(n);`, `ctx.workspace(n).span()` and
    // `.data()` aliased silently before. evidence: docs/design/workspace.md#workspace-lvalue-only-lease-accessors
    Span<std::byte> span() const &;
    Span<std::byte> span() const && = delete;
    /// @brief Implicit conversion to the borrowed bytes; lvalue-only, as span().
    operator Span<std::byte>() const &;
    operator Span<std::byte>() const && = delete;

    /// @brief Pointer to the borrowed bytes; lvalue-only, as span(). `nullptr` for an empty lease.
    std::byte* data() const & { return ptr_; }
    std::byte* data() const && = delete;
    /// @brief Number of bytes borrowed.
    std::size_t size() const { return size_; }

    /// @brief Give the bytes back before the handle goes out of scope.
    ///
    /// Idempotent and never throws, so it is safe from a destructor. Free on an in-order Queue.
    /// On an out-of-order Queue it blocks until the queue is idle when it actually hands bytes
    /// back (when this is the innermost live lease); an out-of-order return only marks the loan.
    /// @warning It waits only on the Queue the lease came from: it does not order against work
    ///          submitted to a Queue derived from it (`Queue(base, in_order)`). Pass your own span
    ///          for work on a sibling queue.
    // evidence: docs/design/workspace.md#workspace-release-on-out-of-order-queues
    void release() noexcept;

private:
    friend struct Queue;
    // false suppresses the arena's debug assert for the one caller that must release out of order.
    void release_(bool diagnose_out_of_order) noexcept;

    WorkspaceLease(Queue* q, std::byte* p, std::size_t n, std::size_t block, std::size_t offset,
                   std::uint64_t seq)
        : queue_(q), ptr_(p), size_(n), block_(block), offset_(offset), seq_(seq) {}

    Queue* queue_ = nullptr;
    std::byte* ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t block_ = 0;
    std::size_t offset_ = 0;
    // Lets the arena tell an innermost release (rewind) from an out-of-order one (do not).
    std::uint64_t seq_ = 0;
};

}  // namespace batchlas

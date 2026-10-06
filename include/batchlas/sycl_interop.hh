#pragma once
/// @file
/// @brief SYCL interop: batchlas::Queue and batchlas::Event to and from `sycl::queue` and
/// `sycl::event`.
///
/// This is the one BatchLAS header that includes `<sycl/sycl.hpp>`, and it is deliberately not
/// reachable from `<batchlas.hh>`. Include it only in the translation units that move a queue or
/// an event across the boundary (they must be compiled with `-fsycl`), and do not include it from
/// a header of your own that the rest of a project pulls in: that puts `sycl.hpp` back into every
/// translation unit.
///
/// Not needed for the native stream (Queue::native_handle() returns it as `void*`) or for memory:
/// USM pointers from `sycl::malloc_device`, `sycl::malloc_host`, `cudaMalloc` or
/// `cudaMallocManaged` wrap zero-copy in a Span or MatrixView, provided they are reachable from
/// the Queue's context. Everything here is single-threaded in the same sense as Queue.
/// @ingroup core
// Kept out of the umbrella for compile time: see the note at the top of blas/linalg.hh.
#include <batchlas/export.hh>
#include <sycl/sycl.hpp>

#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas {

/// @brief The `sycl::queue` BatchLAS submits this Queue's work on.
///
/// A reference to the live queue, not a copy: work submitted to it is ordered against BatchLAS's
/// own work by the queue itself (for the default in-order Queue, after everything already
/// submitted). Its context and device are the ones BatchLAS allocates against, so
/// `get_context()` on it is the right context for USM that BatchLAS should read.
/// @param ctx the Queue; must not be moved-from
/// @return the underlying queue. Do not destroy it, hold it past `ctx`'s lifetime, or submit to
///         it from another thread.
/// @ingroup core
BATCHLAS_API sycl::queue& sycl_queue(const Queue& ctx);

/// @brief The `sycl::event` underlying a BatchLAS Event.
///
/// For handing BatchLAS's work to SYCL code: `depends_on()`, `wait_and_throw()`, a barrier on
/// another queue.
/// @param event the BatchLAS event
/// @return the underlying event, or a default-constructed (already complete) `sycl::event` when
///         `event` is default-constructed or moved-from
/// @ingroup core
BATCHLAS_API sycl::event sycl_event(const Event& event);

/// @brief Wrap a foreign `sycl::event` as a BatchLAS Event, so BatchLAS work can be ordered after
/// work BatchLAS did not submit, with no host synchronisation.
///
/// @code
/// sycl::event mine = my_queue.submit(...);
/// Event e = batchlas::event_from_sycl(mine);
/// ctx.enqueue(e);                  // ctx now runs after mine, no host wait
/// batchlas::potrf(ctx, A, ...);
/// @endcode
/// In the other direction, `sycl_event(ctx.get_event())` hands another queue an event to depend
/// on.
/// @param event the foreign event
/// @return an Event that Queue::enqueue() accepts
/// @pre Both queues live in the same SYCL context.
/// @ingroup core
BATCHLAS_API Event event_from_sycl(sycl::event event);

}  // namespace batchlas

#pragma once
// SYCL interop: BatchLAS's Queue/Event <-> sycl::queue/sycl::event.
//
// This is the one BatchLAS header that speaks SYCL types, and it is deliberately
// unreachable from <batchlas.hh> and <batchlas/blas/linalg.hh>. Those keep <sycl/sycl.hpp>
// out of the umbrella on purpose -- see the note at the top of blas/linalg.hh; it
// is worth ~4.1 s per consumer translation unit -- so nothing in BatchLAS's public
// headers includes this file. Include it yourself, in the few translation units
// that actually have to move a queue or an event across the boundary, and you pay
// for sycl.hpp only there. Do not include it from a header of your own that the
// rest of your project pulls in; that undoes the saving for the whole project.
//
// If all you need is the native stream (cudaStream_t / hipStream_t), you do not
// need this header at all: Queue::native_handle() in <batchlas/util/sycl-device-queue.hh>
// returns it as a void* with no SYCL types involved.
//
// Memory needs nothing from here either. Device pointers from sycl::malloc_device,
// sycl::malloc_host, cudaMalloc or cudaMallocManaged can be wrapped in a Span or a
// MatrixView and used zero-copy, provided they are reachable from the context this
// Queue runs in -- which is what sycl_queue(ctx).get_context() below is for.
//
// Everything here is single-threaded in the same sense the Queue is; see the
// contract note on struct Queue.
#include <sycl/sycl.hpp>

#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas {

// The sycl::queue BatchLAS submits this Queue's work on.
//
// It is a reference to the live queue, not a copy of it, so ordering works the way
// you would expect: work you submit to it is ordered against BatchLAS's own work
// by the queue itself, and for the default in-order Queue that means "after
// everything already submitted". Its context and device are the ones BatchLAS
// allocates against -- get_context() off it is the right context for USM you want
// BatchLAS to read.
//
// Do not destroy it, do not hold it past the Queue's lifetime, and do not submit
// to it from another thread; it is the Queue's queue, not yours. `ctx` must not be
// a moved-from Queue.
sycl::queue& sycl_queue(const Queue& ctx);

// The sycl::event underlying a BatchLAS Event, for handing BatchLAS's work to
// something that speaks SYCL -- depends_on(), sycl::event::wait_and_throw(), your
// own queue's barrier.
//
// A default-constructed or moved-from Event has nothing underneath it; this
// returns a default-constructed sycl::event for that case, which is already
// complete and orders nothing.
sycl::event sycl_event(const Event& event);

// Wrap a foreign sycl::event as a BatchLAS Event, so that BatchLAS work can be
// ordered after work that BatchLAS did not submit.
//
// This is the direction that makes a foreign queue usable without host
// synchronisation, and it is the piece that was missing: Queue::enqueue(Event&)
// already exists and already makes the queue wait for an Event, but until now
// there was no way to build an Event out of an event you had. The pattern is
//
//     sycl::event mine = my_queue.submit(...);
//     Event e = batchlas::event_from_sycl(mine);
//     ctx.enqueue(e);            // ctx now runs after mine, no host wait
//     batchlas::potrf(ctx, A, ...);
//
// and in the other direction sycl_event(ctx.get_event()) hands your queue an event
// to depend on. Both queues must live in the same SYCL context for this to be
// meaningful.
Event event_from_sycl(sycl::event event);

}  // namespace batchlas

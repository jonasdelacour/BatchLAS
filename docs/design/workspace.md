# Workspace: the arena, leases and BumpAllocator sizing {#design_workspace}

> **Covers:** why per-call scratch comes from a per-`Queue` arena through `WorkspaceLease`, the
> lease lifetime rules (nesting, reassignment, out-of-order queues), and how
> `BumpAllocator::measuring()` and `required_bytes()` size a workspace from the same description
> the solve uses, including the alignment trap that makes an exactly simulated size too small.
> **Status:** current. User-facing summary:
> [Workspaces come from the queue's arena](../cpp-api.md#workspaces-come-from-the-queues-arena).
> Implementation: `WorkspaceArena` in `src/queue.hh`, `WorkspaceLease` in
> `src/util/queue-impl.cc`. API reference: the `workspace` group.

## workspace: leases instead of local UnifiedVector

The convenience overloads used to allocate a local `UnifiedVector` for their scratch. A
`UnifiedVector` frees its memory in its destructor, that is, when the calling function returns,
and the kernels using it have only been *enqueued* by then. Every such call site either relied on
an explicit `.wait()` or freed memory out from under work still in flight (`inv`'s
`Matrix`-returning overload did exactly that). A fresh USM allocation per call is also expensive,
and these are per-call scratch buffers whose sizes repeat.

Releasing a lease frees nothing: the memory belongs to the `Queue`, so the pointer stays valid.
Released bytes are handed to the *next* lease, though, so the in-flight question changes from
"freed under running kernels" to "overwritten by later ones". On an in-order queue that is safe by
construction, since the later work is ordered behind the earlier work. On an out-of-order queue
nothing orders them; see
[release on out-of-order queues](#workspace-release-on-out-of-order-queues).

## workspace: lease nesting and release order

Leases nest, and the arena never moves memory that is currently leased: a borrow that does not
fit in the current block opens a new block rather than reallocating, so an outer lease's pointer
stays valid while an inner lease is live. Releasing in reverse order of acquisition, which
scope-bound handles give for free, is the case the arena is built for: the bytes return
immediately. Releasing out of order is safe but wasteful: the arena will not re-serve memory
underneath a live lease, so those bytes stay reserved until the leases taken after them are
released too. Debug builds assert on an out-of-order release so the waste is discoverable. Each
loan carries a sequence number (`seq_`) that the arena uses to tell an innermost release (rewind
the cursor) from an out-of-order one (do not).

A lease is tied to one `Queue` and is not thread-safe, in keeping with `Queue` itself.

## workspace: the reassignment ratchet

Reassigning a live lease is the out-of-order case, unavoidably:

```cpp
ws = ctx.workspace(bigger);   // the right-hand side is acquired before ws is released
```

The new loan is taken on top of the old one, so the old one's bytes cannot be reclaimed until the
new lease dies. In a loop that reassigns every iteration, the arena ratchets: iteration k stacks
a k-th loan rather than reusing the same bytes, and the peak is given back only when the last
lease goes. Call `release()` first when that matters:

```cpp
ws.release();
ws = ctx.workspace(bigger);   // now the old bytes are the ones served
```

The move-assignment operator passes `diagnose_out_of_order = false` to the arena, so this path
does not assert: it is legal to write, and the arena cannot do better. The ratchet is documented
rather than reported at run time.

## workspace: lvalue-only lease accessors

Every accessor that hands out the borrowed memory (`span()`, the conversion to
`Span<std::byte>`, `data()`) is lvalue-only, and the rvalue overload is deleted rather than
absent. A lease is a scope guard: the bytes go back to the arena when it dies and are re-served
to the next borrow, so a pointer taken from a temporary lease is already stale on the next line.
All three of these compiled clean under `-Wall -Wextra -Wdangling` before the change, and all
three aliased:

```cpp
Span<std::byte> ws = ctx.workspace(n);      // lease dies here
auto s = ctx.workspace(n).span();           // and here
std::byte* p = ctx.workspace(n).data();     // and here
```

The correct spelling differs from the first only by `auto`, which is why the compiler has to be
the one to say it. A `const&`-qualified accessor does not do the job (it binds to a prvalue), so
the accessors are ref-qualified `const&` with a deleted `const&&` twin.

## workspace: release on out-of-order queues

`release()` is idempotent and never throws, so it is safe from a destructor. On an in-order queue
it is free. On an out-of-order queue it blocks until the queue is idle *when the release actually
hands bytes back*, that is, when this is the innermost live lease, because the next borrow would
otherwise be written by a kernel the runtime may schedule against work still reading these bytes.
An out-of-order return costs nothing at that point: it only marks the loan, and the later release
that finally reclaims it does the draining. A caller that wants an out-of-order queue and no stall
has to keep the lease alive until it has waited on the work itself, or pass its own span.

Two things the drain does not cover:

- **Derived queues.** It waits on the queue the lease was taken from and nothing else. The
  ops that need ordering (`syev`, `gesvd` and `ormqr` through `select::on_in_order_queue` in
  `src/select/select.hh`, and `iluk` in `src/extensions/iluk.cc`) build a derived in-order
  `Queue` from an out-of-order `ctx` and submit their kernels there, while the lease, taken by the
  convenience overload above them, belongs to `ctx`. Waiting on `ctx` says nothing about work on
  the derived queue; what makes that safe is the derived queue being destroyed, and so drained,
  before the dispatcher returns. In general a lease released on queue X does not order against
  work submitted to a queue derived from X.
- **Blocking convenience overloads.** Scope-bound leases inside the option-struct convenience
  overloads (`blas/options.hh`) are innermost, so on an out-of-order `Queue` those calls block at
  scope exit, where before the arena they returned as soon as the work was enqueued.

## workspace: forward declarations in workspace.hh

`workspace.hh` does not include `util/sycl-span.hh`, because that header includes
`util/sycl-device-queue.hh`, which includes `workspace.hh`. `Span` and `Queue` are forward
declared instead, and the accessors that mention `Span` are defined out of line. Both forward
declarations must stay inside `namespace batchlas`: a forward declaration in another namespace
declares a different type, and the error then surfaces far away, as an incomplete type or a
failed conversion in `blas/options.hh` or `src/util/queue-impl.cc`.

## workspace: BumpAllocator sizing mode

`BumpAllocator::measuring()` returns a pool over a fictitious, maximally aligned, effectively
unbounded region. It runs the same bump arithmetic as a real pool and reports, through
`required_bytes()`, the smallest buffer that satisfies the same call sequence: give a real pool
`required_bytes()` and every allocation succeeds, with at most one alignment quantum to spare.

That holds because every alignment the allocator uses is `max(device_align, alignof(T))`, where
`device_align` is at least 16 and every `T` allocated has `alignof(T) <= 16`. The alignment is
therefore one uniform value, and the layout depends only on offsets relative to the start of the
pool. The fake base (`1 << 32`) is aligned far beyond any device requirement, and a real pool's
base is device-aligned, so both produce the same offsets. The fake extent is
`SIZE_MAX / 4`: large enough that no real request trips the capacity check, small enough that
base plus extent cannot wrap.

The pointers handed out in sizing mode are non-null and correctly aligned, so views can be built
over them, but they address nothing: a dereference faults immediately instead of corrupting
memory. Sizing code must build views over workspace only, never touch their contents, and never
launch a kernel. `remaining()` throws `api_misuse` in sizing mode: its extent is fictitious, so a
callee that sized itself against `remaining().size()` would size against a meaningless number.
Such call sites (`iluk`, `syevx_lobpcg`) were converted deliberately. Because sizing mode hands
out unbacked pointers, never test `ws.data() != nullptr` to mean "the caller passed a
workspace"; use separate overloads.

## workspace: the BumpAllocator alignment trap

`allocate<T>()` checks the alignment-rounded size against the bytes left, measured from the
unaligned cursor, but advances the cursor by the raw extent. A pool sized by the sum of the
advances alone therefore fails its own capacity check on any allocation whose extent is not a
multiple of the alignment: an exactly simulated size is too small. Sizing mode records, per
allocation, the larger of what the check needs and what the data needs, and takes the high-water
mark.

`required_bytes()` also rounds that mark up to the coarsest alignment the sequence asked for.
Sizing results have always been alignment multiples (they were sums of rounded
`allocation_size` terms), and callers depend on it: several add a callee's size straight into
their own total and then re-serve it with `allocate<std::byte>()`, which rounds again. An
unrounded figure silently under-provisions every such caller by up to one quantum.

So: size with `BumpAllocator::measuring()` plus `required_bytes()`, never by re-deriving the sum
by hand, and make a `*_buffer_size` and its run path branch identically.

## workspace: layout functions

An algorithm describes its workspace exactly once, in a `*_layout` function, and both its
`*_buffer_size` entry point and its implementation go through that one description, so neither
can drift from the other:

```cpp
template <Backend B, typename T>
FooWorkspace<T> foo_layout(Queue& ctx, BumpAllocator& pool, /* shape args */) {
    return { pool.allocate<T>(ctx, n * batch), /* ... */ };
}

size_t foo_buffer_size(Queue& ctx, /* ... */) {
    return workspace_bytes([&](BumpAllocator& p) { return foo_layout<B, T>(ctx, p, /* ... */); });
}
```

A layout function must be pure with respect to the workspace: it may read the caller's views
(their shapes and contents are real) and build views over what it allocates, but it must never
read or write workspace memory and never launch a kernel. Nested size queries must be asked about
the caller's views, not about workspace-derived ones, for the same reason.

`workspace_bytes` is the one function in `mempool.hh` re-exported at global scope by the
`BATCHLAS_NO_GLOBAL_NAMES` compatibility shim. Every other unqualified spelling a consumer might
use survives the move into `namespace batchlas` through ADL on its argument, but
`workspace_bytes` takes only a lambda, whose associated namespace is wherever the consumer wrote
it. Without the using-declaration, `workspace_bytes([](auto& s){ ... })` at global scope stops
compiling.

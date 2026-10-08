# Workspace {#design_workspace}

> **Status:** current

Per-call scratch comes from an arena owned by each `Queue`, borrowed through `WorkspaceLease`. Sizing
uses `BumpAllocator::measuring()` and `required_bytes()`, which replay the solve's allocation sequence.
User summary: [Workspaces come from the queue's arena](../cpp-api.md#workspaces-come-from-the-queues-arena).
Implementation: `WorkspaceArena` in `src/queue.hh`, `WorkspaceLease` in `src/util/queue-impl.cc`.

## workspace: leases, not local UnifiedVectors

A local `UnifiedVector` frees on return, while the kernels that use it are only enqueued. A lease frees
nothing. The memory belongs to the `Queue`, so the pointer stays valid, but the next lease reuses released
bytes, so in-flight work can be overwritten. That is safe on an in-order queue. For out-of-order queues see
[release on out-of-order queues](#workspace-release-on-out-of-order-queues).

## workspace: lease nesting and release order

Leases nest, and the arena never moves leased memory. Releasing in reverse acquisition order returns bytes
at once. Out-of-order release is safe but wastes bytes until the later leases are released, and debug
builds assert on it. A lease belongs to one `Queue` and is not thread-safe.

## workspace: the reassignment ratchet

```cpp
ws = ctx.workspace(bigger);   // the new loan stacks on the old one
```

The old bytes are reclaimed only when the new lease dies, so a loop that reassigns stacks one loan per
iteration. Call `ws.release()` first to reuse them. Move assignment does not diagnose this.

## workspace: lvalue-only lease accessors

`span()`, the conversion to `Span<std::byte>`, and `data()` are lvalue-only. The rvalue overloads are
deleted, since a pointer from a temporary lease is stale on the next line. These three compile cleanly under
`-Wall -Wextra -Wdangling` and alias:

```cpp
Span<std::byte> ws = ctx.workspace(n);      // lease dies here
auto s = ctx.workspace(n).span();           // and here
std::byte* p = ctx.workspace(n).data();     // and here
```

The correct spelling adds `auto` to the first line.

## workspace: release on out-of-order queues

`release()` is idempotent and never throws. On an in-order queue it is free. On an out-of-order queue it
blocks until the queue is idle, but only when the release hands bytes back (the innermost live lease).
Otherwise the next borrow could be written while earlier work still reads those bytes. To avoid the stall,
keep the lease alive until you have waited on the work, or pass your own span.

The drain covers only the queue the lease came from:

- **Derived queues.** `syev`, `gesvd`, `ormqr` (`select::on_in_order_queue`, `src/select/select.hh`) and
  `iluk` submit to an in-order queue derived from an out-of-order `ctx`. Waiting on `ctx` does not cover it.
  Safety comes from the derived queue being destroyed, and so drained, before the dispatcher returns.
- **Option-struct overloads** (`blas/options.hh`) hold innermost leases, so on an out-of-order `Queue` those
  calls block at scope exit.

## workspace: forward declarations in workspace.hh

`workspace.hh` cannot include `util/sycl-span.hh`, which includes it back. `Span` and `Queue` are forward
declared inside `namespace batchlas`, and the accessors that name `Span` are defined out of line. A forward
declaration in another namespace names a different type, and the error appears far away.

## workspace: BumpAllocator sizing mode

`BumpAllocator::measuring()` runs the same bump arithmetic over a fictitious, maximally aligned, unbounded
region. `required_bytes()` is the smallest real pool that serves the same sequence, with at most one
alignment quantum to spare. This holds because every alignment is `max(device_align, alignof(T))`, one
uniform value, so the layout depends only on offsets from the pool start.

- Sizing pointers are non-null and aligned. Dereferencing one faults. Sizing code must not touch contents or launch kernels.
- `remaining()` throws `api_misuse` in sizing mode.
- Never test `ws.data() != nullptr` to mean "the caller passed a workspace". Use separate overloads.

## workspace: the BumpAllocator alignment trap

`allocate<T>()` checks the rounded size against the bytes left from the unaligned cursor, but advances the
cursor by the raw extent. Summing the advances therefore undersizes a pool. Sizing mode keeps the larger of
the check's need and the data's need for each allocation, and takes the high-water mark.

`required_bytes()` rounds that mark to the coarsest alignment asked for, because callers add a callee's size
to their own total and re-serve it with `allocate<std::byte>()`, which rounds again. Size with `measuring()`
and `required_bytes()`, never by hand, and make a `*_buffer_size` and its run path branch identically.

## workspace: layout functions

Each algorithm describes its workspace once, in a `*_layout` function that both its `*_buffer_size` and its
implementation call:

```cpp
template <Backend B, typename T>
FooWorkspace<T> foo_layout(Queue& ctx, BumpAllocator& pool, /* shape args */) {
    return { pool.allocate<T>(ctx, n * batch), /* ... */ };
}

size_t foo_buffer_size(Queue& ctx, /* ... */) {
    return workspace_bytes([&](BumpAllocator& p) { return foo_layout<B, T>(ctx, p, /* ... */); });
}
```

A layout function is pure with respect to the workspace. It reads the caller's views but never workspace
memory, and never launches a kernel.

`workspace_bytes` takes only a lambda, so ADL cannot find it from another namespace. The
`BATCHLAS_NO_GLOBAL_NAMES` shim re-exports it at global scope.

# Workspace {#design_workspace}

> **Status:** current · checked 2026-10-21

Per-call scratch comes from an arena owned by each `Queue`, borrowed through `WorkspaceLease`. Sizing
uses `BumpAllocator::measuring()` and `required_bytes()`, which replay the solve's allocation sequence.
User summary: [Workspaces come from the queue's arena](../cpp-api.md#workspaces-come-from-the-queues-arena).
Implementation: `WorkspaceArena` in `src/queue.hh`, `WorkspaceLease` in `include/batchlas/util/workspace.hh`
and `src/util/queue-impl.cc`, `BumpAllocator` in `include/batchlas/util/mempool.hh`; arena internals in
[runtime internals](runtime-internals.md#runtime-internals-the-per-queue-workspace-arena).

## workspace: leases, not local UnifiedVectors

A local `UnifiedVector` frees on return while the kernels that use it are only enqueued. A lease frees
nothing: the memory belongs to the `Queue`, so the pointer stays valid, but the next lease reuses the
released bytes, so in-flight work can be overwritten. That is safe on an in-order queue; for
out-of-order queues see [release on out-of-order queues](#workspace-release-on-out-of-order-queues).

## workspace: lease nesting and release order

Leases nest, and the arena never moves leased memory. Releasing in reverse acquisition order returns
bytes at once. Out-of-order release is safe but holds the bytes until the later leases are released,
and debug builds assert on it. A lease belongs to one `Queue` and is not thread-safe; a wrong-thread
acquire throws `batchlas::api_misuse`, and `Queue::attach_to_current_thread()` transfers a queue with no
outstanding lease.

## workspace: the reassignment ratchet

```cpp
ws = ctx.workspace(bigger);   // the new loan stacks on the old one
```

The old bytes are reclaimed only when the new lease dies, so a loop that reassigns stacks one loan per
iteration. Call `ws.release()` first to reuse them. Move assignment does not diagnose this.

## workspace: lvalue-only lease accessors

`span()`, the conversion to `Span<std::byte>`, and `data()` are lvalue-only; the rvalue overloads are
deleted, because a pointer from a temporary lease is stale on the next line. Without the deletion
these three compile cleanly under `-Wall -Wextra -Wdangling` and alias:

```cpp
Span<std::byte> ws = ctx.workspace(n);      // lease dies here
auto s = ctx.workspace(n).span();           // and here
std::byte* p = ctx.workspace(n).data();     // and here
```

The correct first line is `auto ws = ctx.workspace(n);`.

## workspace: release on out-of-order queues

`release()` is idempotent and never throws. On an in-order queue it is free. On an out-of-order queue
it blocks until the queue is idle, but only when the release hands bytes back (the innermost live
lease); otherwise the next borrow could be written while earlier work still reads those bytes. To avoid
the stall, keep the lease alive until you have waited on the work, or pass your own span.

The drain covers only the queue the lease came from:

- **Derived queues.** `syev`, `gesvd`, `ormqr`, `posv` and the multi-launch `linalg` wrappers run on an
  in-order queue derived from an out-of-order `ctx` (`detail::on_in_order_queue`,
  `<batchlas/util/sycl-device-queue.hh>`; `select::on_in_order_queue` in `src/select/select.hh`). On
  return, `ctx` is ordered after the derived queue's last event, so `ctx.wait()` covers it; a lease on
  `ctx` released *inside* the derived run is not covered. `iluk` waits on its derived queue before it
  returns.
- **Option-struct overloads** (`blas/options.hh`) hold innermost leases, so on an out-of-order `Queue`
  those calls block at scope exit.

## workspace: the arena keeps its peak

The arena grows geometrically and never shrinks until `trim_workspace()` or `~Queue`. An op that leases
internal scratch with no `*_buffer_size` (symm and trmm `expand`, gemm `splitk`) therefore pins its peak
in the `Queue`. Past the device's memory the arena throws `std::bad_alloc` after the choice is made, so
each such family's `can_run` caps its lease:

| Family | Cap |
| --- | --- |
| symm/trmm `expand` (`expansion_fits`, `src/expansion_budget.hh`) | GLOBAL_MEM_SIZE / 4 (lowered by `BATCHLAS_EXPAND_MAX_BYTES`); also batch <= the device's grid z and \f$2^{31}\f$ elements of padded scratch |
| gemm `splitk` | `select::Device::scratch_budget` = min(MAX_MEM_ALLOC_SIZE, GLOBAL_MEM_SIZE / 4); partials are \f$s \cdot ldw \cdot n \cdot batch \cdot \mathrm{sizeof}(T)\f$ ([split-k](../perf/gemm.md#gemm-wp-l-global-split-k-over-the-transposed-wide-tiles)) |

Call `trim_workspace()` after a large call if the memory is wanted back.

## workspace: forward declarations in workspace.hh

`workspace.hh` cannot include `util/sycl-span.hh`, which includes it back. `Span` and `Queue` are
forward declared inside `namespace batchlas`, and the accessors that name `Span` are defined out of
line. A forward declaration in another namespace names a different type, and the error appears far
away.

## workspace: BumpAllocator sizing mode

`BumpAllocator::measuring()` runs the same bump arithmetic over a fictitious, unbounded region based at
\f$2^{32}\f$ (aligned past any `MEM_BASE_ADDR_ALIGN`, unmapped). `required_bytes()` is the smallest real
pool that serves the same sequence, with at most one alignment quantum to spare. This holds because
every alignment is `max(device_align, alignof(T))`, one uniform value, so the layout depends only on
offsets from the pool start.

- Sizing pointers are non-null and aligned; dereferencing one faults. Sizing code must not touch
  contents or launch kernels.
- `remaining()` on a sizing pool and `required_bytes()` on a real pool throw `batchlas::api_misuse`. A
  real pool that is too small throws `batchlas::workspace_error`.
- Never test `ws.data() != nullptr` to mean "the caller passed a workspace". Use separate overloads.

## workspace: the BumpAllocator alignment trap

`allocate<T>()` checks the rounded size against the bytes left from the unaligned cursor, but advances
the cursor by the raw extent, so summing the advances undersizes a pool. Sizing mode keeps the larger
of the check's need and the data's need for each allocation and takes the high-water mark.

`required_bytes()` rounds that mark to the coarsest alignment requested, because callers add a callee's
size to their own total and re-serve it with `allocate<std::byte>()`, which rounds again. Size with
`measuring()` and `required_bytes()`, never by hand, and make a `*_buffer_size` and its run path branch
identically.

## workspace: layout functions

Each algorithm describes its workspace once, in a `*_layout` function that both its `*_buffer_size`
and its implementation call:

```cpp
template <Backend B, typename T>
FooWorkspace<T> foo_layout(Queue& ctx, BumpAllocator& pool, /* shape args */) {
    return { pool.allocate<T>(ctx, n * batch), /* ... */ };
}

size_t foo_buffer_size(Queue& ctx, /* ... */) {
    return workspace_bytes([&](BumpAllocator& p) { return foo_layout<B, T>(ctx, p, /* ... */); });
}
```

A layout function is pure with respect to the workspace: it reads the caller's views but never
workspace memory, never launches a kernel, and asks nested size queries about the caller's views only.

`workspace_bytes` takes only a lambda, so ADL cannot find it from another namespace; the
`BATCHLAS_NO_GLOBAL_NAMES` shim re-exports it (and `BumpAllocator`) at global scope.

# ILU(k) {#perf_iluk}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda. The figures below have no
> recorded date, harness or raw file.

Batched ILU(k) in `src/extensions/iluk.cc` runs its numeric phase on the host or the device. The
choice depends on batch size alone. All items share one sparsity pattern: pattern-only work runs
once on the host, and only the per-item arithmetic scales with the batch. ILU(k) has no flat
selection op, no tuned table and no `BATCHLAS_ILUK_ROUTE` (see
[flat kernel selection](../design/flat-kernel-selection.md)).

## iluk: the host-versus-device crossover

- `iluk_prefer_device(batch_size)` returns `batch_size >= 32` unless `BATCHLAS_ILUK_DEVICE` pins
  it. The variable reads only the first character: `0` forces host, `1` forces device, and
  `true`/`on`/`yes` fall through to the batch default. `iluk_tests` uses it to check the two paths
  against each other.
- The device path has a fixed cost per call: one launch per dependency level plus host round
  trips. The host path scales with the batch.
- On a 2D Laplacian the two cross at roughly 32 batch elements. Re-measure before moving the
  threshold.

## iluk: host numeric-phase cost findings

Each finding was the limiting cost at some point, and each explains part of the host path.

- **The batch loop is parallel.** Each item reads and writes only its own slice. At batch 1024 it
  dominated the rest of the solver.
- **No allocation in the inner loop.** A per-row heap allocation of the fill-candidate list
  contended on the allocator once the loop was threaded. `candidates` is now caller-owned scratch.
  The symbolic pattern and per-item values are flat CSR-style arrays shared by the batch, not
  `vector<vector<...>>` (`n * batch_size` allocations). `col_to_slot` (column to slot, `-1` outside
  the pattern) gives an O(1) lookup in place of a binary search.
- **No re-derived row scale.** Row `j < i` is final when row `i` eliminates against it. Re-deriving
  its scale would be an O(nnz_j) scan that recomputes a stored value.
- **Sizing reads item 0 only.** `check_batch_sparsity` walks every item's pattern, which on unified
  memory means a page migration per item. `iluk_buffer_size` sizes against item 0. The size is an
  upper bound, because the numeric phase only prunes.
- **Deterministic fill selection.** Sort ties break by column, so the surviving pattern depends only
  on the input. Host and device then agree.

## iluk: the device factorization

- **Pattern-only indices on the host.** The elimination schedule and both level schedules depend
  only on the pattern. They are built once for any batch size, and the device does pure arithmetic.
  The schedule applies updates from dropped sources too. Dropped entries are zero, so these are
  no-ops, and the schedule stays independent of the values.
- **One packed integer buffer.** One managed allocation per symbolic array dominated the cost of
  factorizing a single small system. The arrays are packed into one allocation. Its tail holds the
  compaction scratch and status flags.
- **Pattern agreement is checked on the device.** A host check would read every item's indices
  across the bus.
- **Slot-major values** `work[slot * batch + b]`. Adjacent work-items differ in `b`, so their
  accesses are contiguous. The final gather transposes into the batch-major CSR layout that the
  apply kernel reads.
- **One launch per dependency level, on an in-order queue.** The launch count is the factor depth.
  Waiting on the host between levels cost one round trip per level: 53 for a 4096-row ILU(1)
  factor. The levels are submitted to an in-order queue instead. The queue is held in a
  `std::optional<Queue>`, because constructing a `Queue` builds a real `sycl::queue` and the common
  in-order path needs no second one.
- **Kernel-side pivot failure** sets a status flag, because a kernel cannot throw. The flag is
  checked once after the queue drains.
- **Fill quota in the kernel.** Repeatedly dropping the smallest survivor equals sort-and-truncate,
  with no per-row scratch. With the default `fill_factor` the loop does not run.
- **Compaction** keeps an entry if any item kept it. A device reduction over the batch produces the
  flags, so the host scans `sym_nnz` flags, a count independent of batch size.

## iluk: the apply kernel

One work-group handles each (batch item, right-hand-side column) pair. Rows of a dependency level
are split across work-items, with a group barrier between levels. The serial chain is therefore the
level count, not `n`, and no cross-group synchronisation is needed. Values and column indices are
replicated per item, because the kernel indexes them with the same matrix stride.

## iluk: open debts

- The ~32-item crossover has no recorded grid. A saturating sweep over batch size and at least two
  patterns (a 2D Laplacian and a denser one) would show whether the threshold should also depend on
  `n` or on the level count.

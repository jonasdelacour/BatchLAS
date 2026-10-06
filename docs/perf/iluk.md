# ILU(k): the host and device numeric paths {#perf_iluk}

> **Covers:** batched ILU(k) (`src/extensions/iluk.cc`): which numeric path runs, the cost
> findings that shaped the host and device factorizations, and the apply kernel.
> **Status:** current. Distilled 2026-09-30 from the source comments it replaces.
> **Machine:** RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda, as far as the source records;
> the original comments give no date, harness or raw file for any figure below.

ILU(k) has no `RouteTable` (`Op::iluk` exists but only for `op_name`, see
[vendor independence](../design/vendor-independence.md)); the one routing decision is host versus
device for the numeric phase, taken on batch size alone. Every batch item shares one sparsity
pattern, and both paths exploit that: everything that depends only on the pattern is computed
once on the host, and only the per-item arithmetic scales with the batch.

## iluk: the host-versus-device crossover

`iluk_prefer_device(batch_size)` (`src/extensions/iluk.cc`) returns `batch_size >= 32` unless
`BATCHLAS_ILUK_DEVICE` pins it.

- The device path has a fixed cost per call (one kernel launch per dependency level plus a few
  host round trips) that does not shrink for small problems; the host path costs time
  proportional to the batch.
- **Measured on a 2D Laplacian, the two cross at roughly 32 batch elements.** No grid, date or
  raw file for this measurement survives in the tree; re-measure before moving the threshold.
- `BATCHLAS_ILUK_DEVICE` inspects only the first character (`0` forces host, `1` forces device;
  `true`/`on`/`yes` fall through to the batch default, narrower than `env_truthy` on purpose). It
  is also how the two paths are checked against each other in `iluk_tests`.

## iluk: host numeric-phase cost findings

Each of these was the limiting cost at some point and is why the host path looks the way it does.

- **The batch loop is parallel.** Batch items share a pattern but nothing else: each item's numeric
  phase reads only its own slice of `A` and writes only its own slice of the values, so the loop is
  embarrassingly parallel. It is also the part that scales with the batch: **at batch 1024 it
  dominated everything else the solver did.**
- **No allocation in the inner loop.** Once the batch loop ran on several threads, a per-row heap
  allocation of the fill-candidate list became the limiting factor (all threads contend on one
  allocator), so `candidates` is caller-owned scratch reused across rows. Likewise the symbolic
  pattern and per-item values used to be `vector<vector<...>>`, one allocation per row per item
  (`n * batch_size` of them), invisible at batch 1 and dominant once the loop is threaded; they are
  now flattened into CSR-style arrays shared by the batch. `col_to_slot` (column to slot in the
  row being assembled, `-1` outside the pattern) gives an O(1) lookup in place of a binary search
  per touched entry.
- **No re-derived row scale.** Row `j < i` is final when row `i` eliminates against it: its
  diagonal was stabilised when it was processed and stabilisation is idempotent, so re-deriving
  the scale (an O(nnz_j) scan per elimination step) would recompute a stored value.
- **Sizing skips the full pattern walk.** `check_batch_sparsity` walks every item's pattern, which
  on unified memory the device has just written means a page migration per item.
  `iluk_buffer_size` sizes against item 0 only, and the factorization validates before it uses
  anything. The size is an upper bound because the numeric phase only prunes (drop tolerance,
  fill quota).
- **Deterministic fill selection.** Ties in the fill-candidate sort break by column so the
  surviving pattern is a function of the input alone; `std::sort`'s unspecified order would make
  the factor depend on the library implementation and would stop host and device agreeing.

## iluk: the device factorization

- **Pattern-only indices on the host.** Which slot each elimination reads and updates, and which
  rows may run concurrently, depend only on the pattern, so the elimination schedule and both
  level schedules are built once on the host whatever the batch size; the device does pure
  arithmetic over each item's values. The schedule applies updates whose source was dropped too
  (dropped entries are zero, so they are no-ops), which keeps it independent of the values.
- **One packed integer buffer.** The symbolic index arrays are small but numerous, and **one
  managed allocation each dominated the cost of factorizing a single small system.** They are
  packed into one allocation, whose tail also holds the compaction scratch and status flags.
- **Pattern agreement is checked on the device.** Checking it on the host means reading every
  item's indices back across the bus, the batch-proportional host work this path exists to avoid.
- **Slot-major working values** (`work[slot * batch + b]`): one work-item handles one
  (row, item) pair, so adjacent work-items differ in `b`, and slot-major makes their accesses
  contiguous. The final gather transposes into the CSR batch-major layout the apply kernel reads.
- **One launch per dependency level, ordered by the queue.** The launch count is the factor's
  depth, a property of the pattern, so the batch only widens each launch. Levels must run in
  order, and waiting on the host between them cost one round trip per level (**53 of them for a
  4096-row ILU(1) factor**), so the levels are submitted to an in-order queue instead. It is held
  in a `std::optional<Queue>` because constructing a `Queue` builds a real `sycl::queue`, and the
  common in-order path never needs a second one.
- **Kernel-side pivot failure** is reported through a status flag rather than an exception (a
  kernel cannot throw to the host) and rechecked once after the queue drains.
- **Fill quota in the kernel.** Repeatedly dropping the smallest survivor is equivalent to sort
  and truncate and needs no per-row scratch; with the default `fill_factor` the loop does not run.
- **Compaction reduces over the batch on the device**: an entry survives if any item kept it, so
  the host scans `sym_nnz` flags, a figure independent of the batch size.

## iluk: the apply kernel

One work-group per (batch item, right-hand-side column) system. Within a group the rows of a
dependency level are split across work-items with a group barrier between levels, so the serial
chain is the level count rather than `n`, and no cross-group synchronisation is needed. The
pattern is identical across the batch, but the kernel indexes values and column indices with the
same matrix stride, so both are replicated per item rather than shared.

## iluk: open debts

- The ~32-item crossover has no recorded grid, machine date or raw file. A saturating sweep over
  batch size and at least two patterns (2D Laplacian and a denser one) would settle whether the
  threshold should depend on `n` or the level count as well as the batch.

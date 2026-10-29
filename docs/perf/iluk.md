# ILU(k) {#perf_iluk}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda; figures have no recorded date
> or raw file.

`src/ops/iluk/iluk.cc` runs the numeric phase on the host or the device, by batch size alone. All
items share one pattern: pattern-only work runs once on the host. No flat selection op, tuned table
or `BATCHLAS_ILUK_ROUTE` (see [flat kernel selection](../design/flat-kernel-selection.md)).

## iluk: the host-versus-device crossover

`iluk_prefer_device(batch_size)` returns `batch_size >= 32`. `BATCHLAS_ILUK_DEVICE` reads only the
first character (`0` host, `1` device; anything else falls through); `iluk_tests` uses it to compare
the paths. The device path pays one launch per dependency level; the host path scales with the batch.
On a 2D Laplacian they cross near 32 items. Re-measure before moving the threshold.

## iluk: host numeric-phase cost findings

- The batch loop is parallel; each item touches only its own slice.
- No allocation in the inner loop: `candidates` is caller-owned scratch; pattern and values are flat
  CSR-style arrays; `col_to_slot` (`-1` outside the pattern) replaces a binary search.
- Row `j < i` is final when row `i` eliminates against it, so its scale is read, not re-derived.
- `iluk_buffer_size` sizes against item 0 only (an upper bound, since the numeric phase only prunes);
  walking every item would page-migrate per item on unified memory.
- Sort ties break by column, so host and device keep the same fill.

## iluk: the device factorization

- Elimination and level schedules are built once on the host from the pattern; the device does pure
  arithmetic. Updates from dropped sources are kept as zero no-ops so the schedule stays value-free.
- All symbolic arrays live in one packed managed allocation, whose tail holds compaction scratch and
  status flags. Pattern agreement is checked on the device.
- Values are slot-major (`work[slot * batch + b]`) for coalescing; a final gather transposes to the
  batch-major CSR the apply kernel reads.
- One launch per level on an in-order queue (a host wait per level cost 53 round trips for a
  4096-row ILU(1)). The queue is a `std::optional<Queue>` so the common path builds no second queue.
- Pivot failure sets a status flag, checked once after the queue drains.
- The fill quota drops the smallest survivor repeatedly (no per-row scratch); with the default
  `fill_factor` it does not run. Compaction keeps an entry if any item kept it.

## iluk: the apply kernel

One work-group per (item, right-hand side); rows of a level are split across work-items with a
barrier between levels, so the serial chain is the level count, not `n`.

## iluk: open debts

- The ~32-item crossover has no recorded grid. Sweep batch size on a 2D Laplacian and a denser
  pattern to see whether the threshold should depend on `n` or the level count.

# Masked batches {#design_masked_batches}

> **Status:** current · design choice of P2.6 (the fullerenes plan), revised after review 2026-10-26; nothing timed (RTX 4090 sm_89)

An SCC loop solves one eigenproblem per molecule per iteration, and the molecules converge at
different iterations: a C300 isomer set takes 36 to 75 iterations (median 56). The caller
wants to re-solve only the items still iterating and keep its batch as it is. It should not have
to compact the batch, re-key its arrays and scatter back.

## The design {#masked-batches-design}

- **Entry points.** `syev_masked`, `sygvd_masked` and `sygvd_factored_masked` are the unmasked calls
  plus a trailing `Span<const uint8_t> mask`, one entry per item, nonzero meaning active. They are
  separate functions with their own `*_buffer_size`, not a new parameter on `syev`. They live in
  their own TUs (`src/ops/syev/syev_masked.cc`, `src/ops/sygvd/sygvd_masked.cc`) and call only the
  public unmasked entry points, so `syev.cc`, which the syev and sygvd tuner specs hash, is untouched. Two reasons. The `sig::` aliases are function types and cannot default a parameter. And
  `syev`'s variadic queue-dispatch overload would make an overloaded `syev` ambiguous
  (@ref design_api_conventions). `info` keeps its place, so the mask sits where per-item status
  already sits.
- **Contract.** An inactive item's operands (A, B or the factor), eigenvalues and `info` are not
  touched: they are bit-identical on return. An active item gets what the unmasked call gives it. The
  mask is read on the host before anything is enqueued, so it must be host-readable (USM shared or
  host memory), and the work that wrote it must be complete. A mask shorter than the batch throws
  `batchlas::invalid_argument`.
- **Mechanism: compaction in the workspace.** The active items are gathered into a packed copy in
  the workspace, laid out like the caller's batch (same `ld` and stride), so a family that branches
  on layout takes the same branch. The public unmasked call then runs on that k-item copy and selects
  its family for k items (pins included), and the results are scattered back
  (`src/kernels/common/batch_mask.cc`, one work-item per element). With every item active the call is
  the unmasked one. With none active it is a no-op and reads no workspace.
- **Routing by the active count, not the batch.** The first version chose the family for the whole
  batch and ran it on the copy. Routing by k instead makes the masked call *be* the unmasked call on
  the active items, so the bit-identity of part 1 below needs no pin. It also routes a nearly
  converged batch like the small batch it now is. A table row keyed on k can then pick another family
  than the whole batch would.
- **One workspace size for every mask.** `*_masked_buffer_size` returns the largest need over every
  count of active items: the unmasked size for k = batch, and the packed layout plus the packed call's
  own size for each k from 1 to batch - 1. The mask's contents do not change it; its length is still
  checked. An SCC loop queries it once per geometry and reuses the workspace as items drop out. An
  exact per-mask size would grow when one item of N drops out (the copy adds O(k n^2) bytes while the
  child need barely shrinks), so a size queried with every item active would be too small on the next
  iteration. The cost is one buffer-size query of the unmasked op per item count, all on the host.
- **Cost.** No kernel runs on an inactive item. The overhead is one host-to-device copy of the index
  list (waited on), one gather of the active operands, one scatter of the results, and a workspace
  holding a copy of the active items: O(k n^2) bytes against the O(k n^3) solve.
- **Vendor families.** Compaction serves them as it serves the native ones: the per-item loops
  (cuSOLVER `?sygvd`, LAPACKE) and the batched one (cuSOLVER `XsyevBatched` on a packed batch) see
  only the active items. No family is refused for a masked call.
- **Rejected: an in-kernel early exit.** Gating every work-group on `mask[b]` would avoid the copy.
  But it means threading a mask through every syev pipeline: sytrd, sb2st, stedc, the CTA solvers,
  the back-transforms and the children's own launches. Vendor calls cannot take a mask at all. The
  copy is O(n^2) per active item; the kernel surgery is all of syev.

## Bit-identity with the unmasked call {#masked-batches-bit-identity}

The contract has two parts.

1. **Always:** an active item's result is bit-identical to the unmasked call on the active items
   alone, packed in the same layout: the masked call is that call.
2. **Per-item families:** the result is also bit-identical to the unmasked call on the whole batch.
   These are the families whose per-item result depends only on the item and its layout: syev's
   `cta`, `cta_fused`, `jacobi` and `vendor`, sygvd's `vendor`, and every LAPACKE path.

`blocked`, `two_stage` and `compose` do not get part 2 in general. Their children select by batch
size: the `potrf`, `gemm` and sytrd stages inside them are keyed on the batch they see, which is k,
not the full batch. Their launch shapes are sized by it too. So their results equal the whole-batch
call's only within the verify bound, and their eigenvectors satisfy the residual check.

Measured on sm_89 (2026-10-25, `masked_batch_tests` before part 2 was split out): a single active
item out of 9 differed in bits from the 9-item call for `blocked` and `two_stage` at n = 12 and 40,
and for `compose` at n = 40 (B's factor differed, so `potrf` took another kernel at batch 1). Four
active items out of 9, and 512 out of 1024, were bit-identical on every family. The test now pins
the 512-of-1024 case on every family: it holds because k = 512 and 1024 select the same children in
today's tables, so a table change that splits them turns it red by design, and the fix then is to
move the assertion, not the contract.

A thread-local "select as for batch N" override in `select::` would fix the routing part. It would
not fix the launch sizing, so it was not built.

## What the tests pin {#masked-batches-tests}

`tests/masked_batch_tests.cc` checks:

- the patterns none, all, alternating and single, on every family the backend can pin (syev:
  `cta`, `cta_fused`, `jacobi`, `blocked`, `two_stage`, `vendor`; sygvd: `compose`, `vendor`; the
  factored entry `compose`), both triangles, padded `ld` and stride, and a saturating batch of 1024,
  half masked;
- inactive items poisoned, and their storage, eigenvalues and `info` sentinels bit-identical on return;
- active items bit-identical to the unmasked call on the active items alone (same pin), with
  \f$ \|A X - B X W\| \f$ within the eigen-residual bound (a reference product on the operands before
  the call), and bit-identical to the whole-batch call on a per-item family and, on every family, in
  the 1024 batch half masked (eigenvalues within the values bound otherwise);
- the buffer size is the same for the pattern's mask and the all-active mask, and every pattern runs
  in the all-active size (none in a zero-byte workspace);
- potrf pinned `native` on GPU backends, and the factored entry's B re-poisoned after potrf, so B's
  other triangle holds poison when the code under test runs;
- a short mask is refused.

The deliberate breaks and their red sets are named above the tests.

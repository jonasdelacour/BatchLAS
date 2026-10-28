# Masked batches {#design_masked_batches}

> **Status:** current · the batch mask as a view property; W0 (foundation) of its migration, 2026-10-28. Replaces the compaction design of the closed #207 (`syev_masked`, `sygvd_masked`), which never reached main. Guard cost prototyped on sm_89 only.

An SCC loop solves one eigenproblem per molecule per iteration, and the molecules converge at
different iterations: a C300 isomer set takes 36 to 75 iterations (median 56). The caller wants to
re-solve only the items still iterating and keep its batch as it is, without compacting the batch,
re-keying its arrays and scattering back. The answer is a per-item mask on the view itself, which
every op honours natively, the same way it honours per-item active dimensions
(@ref design_matrix_model).

Three decisions are fixed:

1. Every kernel skips its inactive items itself. Nothing compacts the batch centrally.
2. The mask lives on the view and may be device-only. The library never reads it on the host.
3. `syev_masked` and `sygvd_masked` are gone; the call is `syev(A.with_batch_mask(m), ...)`.

## The design {#masked-batches-design}

- **Representation.** `Span<const std::uint8_t>`, one byte per item, nonzero meaning active. A
  bitset was rejected: a device producer (a convergence kernel) would need atomics, and every
  kernel check a shift.
- **On a view.** `MatrixView` (dense and CSR) and `VectorView` carry a non-owning `batch_mask_`;
  empty means unmasked.
  - `with_batch_mask(mask)` checks only the span's length (empty or `batch_size()`), never its
    contents, and throws `batchlas::invalid_argument` otherwise. An empty span clears the mask.
  - `batch_mask()` and `has_batch_mask()` read it back.
- **In a kernel.** `KernelMatrixView` has `const std::uint8_t* mask_` (null when unmasked) and
  `active(b)`, which is `mask_ == nullptr || mask_[b] != 0`. A device `batch_item(b)` points the
  mask at item b, so `batch_item(b).active(0) == active(b)`. `VectorView::active(b)` is the same
  test.
- **Owned by a Matrix.** `Matrix::enable_batch_mask()` allocates a `UnifiedVector<uint8_t>`, sets
  every byte to 1, binds it, and returns it writable, so a device kernel can write it directly.
  Copy (deep), move and `clone()` carry the store; `view()` attaches it. Caller-owned device memory
  (`sycl::malloc_device`) goes through `with_batch_mask` instead.
- **Propagation**, on the host and without reading the bytes:
  - Carried by `batch_range` (a subspan), the host `batch_item` and `operator[]` (one byte), the
    `operator()` slices, the row and column vector extractors, `with_active_dims`,
    `MatrixView(VectorView)`, `astype<U>()` and `kernel_view()`.
  - The override constructor `MatrixView(view, rows, cols, ld, stride, batch)` keeps the mask, and
    throws when a masked view's batch would change: dropping the mask would run inactive items.
  - The raw-pointer constructors start unmasked. That is the drop hazard: an op that rebuilds a
    view from `data_ptr()` must re-attach `mask.subspan(b0, count)` itself.
- **One mask per call.** `detail::call_mask(op, views...)` (`src/util/batch_mask.hh`) returns the
  one span the masked operands share, or empty. The call batch is the largest operand batch, and
  unmasked operands adopt the mask. Two different spans (by `data()` or `size()`) throw
  `invalid_argument("<op>: operands carry different batch masks")`. A masked broadcast operand
  (another batch, or stride 0 at batch > 1) throws too.
- **Per-item spans** (`info`, eigenvalues, singular values, `tau`, pivots) are indexed by item and
  follow the call mask. `info_clear` and `info_from_flags` (`src/kernels/common/info_span.hh`) take
  the mask and leave an inactive slot untouched.
- **Utilities honour it now:** every `fill*`, `triangularize`, `symmetrize`, `hermitize`, `scale`,
  `MatrixView::copy` and `VectorView::copy`, `all_close` (an inactive item compares true), the
  `linalg` elementwise helpers and `transpose`. Masked, they skip their memset and memcpy fast
  paths. `fill_random_sparse_hermitian` throws `batchlas::unsupported`, because it builds the row
  offsets on the host.

## The kernel contract {#masked-batches-kernel-contract}

**An inactive item's outputs are bit-for-bit untouched:** its matrices, `w`, `info`, pivots, `tau`,
and C even when `beta = 0`. Its scratch is unspecified, and no active item may read it.

Most kernels take raw pointers, not a `KernelMatrixView`, so a migrated family passes the mask as
an explicit kernel argument. Four idioms cover the kernels:

- **I1, one item per work-group** (grid z, or G work-groups per item): return on
  `!active(b)` before the first barrier or group collective. The return is uniform per work-group.
- **I2, several items per work-group or sub-group** (the tiny and CTA kernels, `lpanel`, gemm
  `small`, trsm `sg_left`): never return early. Fold `active(prob)` into the existing `live`
  predicate, load an identity pad for a dead item as a dead tail partition does today, run every
  collective on every lane, and store only when live.
- **I3, flat or grid-stride loops:** `continue` on the item of each index.
- **I4, host-task loops** (NETLIB): read the mask inside the `host_task`. The in-order queue orders
  it after the mask's producer, and CPU USM is host-addressable there.

The prototype on sm_89 (branch `fz/mask-proto`) found the I1 return free and the I2 fold 2-4.5%
slower on float `potrf_tiny`. Packed families therefore get a `template <bool Masked>` instance in
their wave, not in W0.

Placement traps: jacobi checks after its first work-group barrier; split-k's slice and reduce
kernels skip by item (`z / s`), not by z; holes inside a sub-group are safe on NVPTX and untested on
SPIR-V and AMDGCN, where today's tails only ever drop trailing partitions. `gebrd_cta` and
`ormqx_cta` already return per partition before a barrier (undefined when the batch is not a
multiple of the partitions per work-group); both move to I2 in their waves.

## Refusal policy {#masked-batches-refusal-policy}

The precedent is per-item active dimensions: when ops ignored them, 12 ops computed on padded
storage with no error ([Defect 12](known-defects.md#defect-12-vendor-families-accepted-a-heterogeneous-batch)).
The mask follows the fix from the start: **an op family that does not honour the mask refuses a
masked view**, so no op can silently ignore one at any point of the migration.

- W0 adds `!has_batch_mask()` to every family's `can_run` (a correctness term, never a speed one),
  and an entry refusal to the ops without selection (hemm, herk, her2k, lanczos, ortho, syevx,
  norm, cond, inv, iluk, ritz_values). A wave lifts the term family by family once the family
  honours the mask.
- **Vendor families refuse a masked view permanently.** Strided and pointer-array batched calls
  cannot skip an item, and a per-item host loop would need a host read. Auto walks on to a native
  family; a pin on the vendor throws "cannot run this shape". A shape or backend with only a
  vendor route gets the explicit `NoRouteError` or "no runnable kernel".
- **Routing keys stay on `batch_size()`, the capacity.** The active count is unknowable on the host.
  So a masked call on a pin takes that pin's route and launch shapes; `BATCHLAS_SELECT_TRACE` adds
  `masked=1`, and no table key is added.
- **`<op>_buffer_size`** is the chosen family's need, independent of the mask's contents, so an
  SCC loop queries it once. A masked view can choose another family than the unmasked one (the
  vendor is refused), so query with the masked view; a mismatch hits the existing
  `workspace_error` check. stedc adds its per-level expanded mask.
- **Host readbacks** of cross-item values (stedc's `n_reduced` maximum, the syevx and lobpcg
  convergence counts, the gesvd tiny-singular-value fold, the bdsqr failure fold, `all_close`) stay
  where they are: the device kernel that produces the value writes a neutral one for an inactive
  item (0, converged, false, true), so no extra sync is needed.
- A masked call cannot short-circuit: with zero active items it still pays every launch and host
  sync. A host fast path for all-active or all-inactive masks would need a host read.

## Bit-identity with the unmasked call {#masked-batches-bit-identity}

Because routing keys on the capacity, an active item's result is **bit-identical to the unmasked
whole-batch call on the same pinned route**. This holds per pinned route only: Auto may route a
masked call elsewhere, since it refuses the vendor families. The exceptions, checked within the
`EXPECT_VERIFY` bound instead:

- spmm accumulates with atomics, so it is never bit-identical, masked or not;
- the cross-item host decisions above: stedc's deflation maximum at n >= 512, the gesvd
  tiny-singular-value path, the bdsqr fold, and the syevx iteration counts. Bit-identity there is a
  hypothesis that each wave's tests decide.

## Migration waves {#masked-batches-waves}

Each wave after W0 touches disjoint op directories and lifts the refusal of the families it
migrates. The only shared file is the kernel hash pin map, regenerated with
`scripts/agent/repin_tune_hashes.py`.

| Wave | Content |
| --- | --- |
| W0 | the view mask, `call_mask`, the utilities, masked `info_clear`; a refusal in every family; the cross-op tests and `tests/masked_contract.hh` |
| W1a | gemm (every native family, split-k, `small`, the k = 0 path, the heterogeneous loop), gemv |
| W1b | trsm, trmm, symm/hemm, syrk/herk, syr2k/her2k, the level-3 batch helpers, spmm |
| W2a | potrf (tiny, cta, lpanel, blocked), potrs, posv, potri, lascl |
| W2b | getrf, laswp, getrs, gesv, getri, inv |
| W2c | geqrf, larft, orgqr, ormqr, `ormqx_cta` (fixes its partition return) |
| W3a | steqr, stedc (per-level expanded mask, neutral `n_reduced`), stebz, stein |
| W3b | sytrd (cta, blocked, sy2sb, sb2st), every syev family, syev_align, syfn |
| W3c | sygst, sygvd |
| W4a | gesvd (gebrd including the `gebrd_cta` fix, bdsqr, bdsdc, ormbr, gesvdj) |
| W4b | sytrf, sytrs, sysv |
| W4c | ortho, lanczos, syevx, norm, cond, iluk, ritz_values |

Order: W0, W1a, W1b, then W2a, W2b and W2c, then W3a, W3b, W3c, then W4a, W4b and W4c. The SCC path
(syev and sygvd masked) is W0, W1a, W1b, W2a, W2c, W3a, W3b, W3c. A composite keeps refusing until
every child it calls is migrated, because the child runs on the composite's masked views.

## What the tests pin {#masked-batches-tests}

- **The view and the utilities** (`tests/batch_mask_view_tests.cc`): propagation through every
  transform, `call_mask`, and every utility on a device-only mask (inactive items memcmp-equal to
  the poisoned input, active ones to the unmasked call, an all-ones mask bit-identical).
- **Cross-op** (`tests/select_tests.cc`, `BatchPolicy` on `OpBattery`: n = 8, batch 4, padded `ld`,
  stride and `inc`, complex data):
  - `MaskedBatchNeverRunsAFamilyThatIgnoresIt`: every op on Auto and on every compiled candidate
    pinned strictly. A pin must first run unmasked, so a shape refusal cannot pass as a mask
    refusal. With a device-only mask {0, 1, 0, 1} the call is then refused (`unsupported`, the
    pin's "cannot run this shape", or a strict `vendor` pin finding no candidate) or leaves every
    output of items 0 and 2 bit-identical. Each op needs at least one pin that ran.
  - `MaskedOperandsMustAgree`: A carries one span, the other operands another span with the same
    bytes; the call throws and writes nothing. An op that accepts one shared mask must say
    "operands carry different batch masks".
  - `MaskLengthIsChecked`: batch - 1 and batch + 1 throw on dense, heterogeneous, vector, CSR and
    owned views; the exact length binds, an empty span clears, `batch_range` slices.
- **Per op and family** (`tests/masked_contract.hh`, used by each wave; its own checks are in
  `tests/masked_contract_tests.cc`): `masked_contract::check<Choice>(q, op, spelling, case)` pins
  the family strictly and requires a coverage `reached` row of it on a masked call. Then for each
  pattern it checks that every inactive item's outputs (the case lists matrices, `w`, `info`,
  pivots, `tau`) are bit-identical to the poisoned input, and every active item to the unmasked
  call on the same pin, or the case's `verify_active` hook for the exceptions above. The patterns
  are none, all active, alternating, single, the packed work-group holes (slot 0 off, only the last
  slot on, the interior off), and, under `BATCHLAS_TUNE_L2_CHUNK_ITEMS`, the part edges off and one
  part wholly inactive. Every mask is device-only, written by a kernel. `buffer_size` must not move
  with the pattern, and each pattern runs in exactly that size. `identical_items` adds the
  saturating-batch check (active items bit-identical to the first active one).

Deliberate breaks, one per axis (agent guide section 8, rule 1):

- **On the foundation commit, where nothing refuses yet**, `MaskedBatchNeverRunsAFamilyThatIgnoresIt`
  is red on every pin that runs (sm_89 float: 107 of 107; NETLIB float: 40 of 40), naming the pin
  and the inactive item written: that is the red a family ignoring the mask gives. With the
  level-3, eigen and misc refusals merged, their ops went green; dropping gemm's refusal for
  `direct` alone then turned exactly `gemm` pinned `direct` and `gemm` on Auto red on CUDA (both
  scalar types), and nothing on NETLIB, where `direct` does not run (2026-10-28).
- The contract's own checks name their red sets: a call that ignores the mask is red on exactly
  the seven patterns with an inactive item; an active item that differs is red on every masked
  pattern; a size that follows the mask is red on every pattern after the first masked one.
- Per wave: drop `active()` in one family (red: that pin's patterns with holes); stop slicing the
  mask in `batch_range` (red: the chunk patterns); drop one vendor refusal (red: `BatchPolicy` on
  that vendor); revert `info_clear` to a memset (red: the `info` sentinel only).
- **W1a, gemm** (`tests/gemm_mask_tests.cc`; sm_89, every scalar, 2026-10-28): every native family
  runs a masked batch and the vendor refuses. Defeating one guard at a time turned red only its own
  cases: `direct`'s I1 return (the `direct` pins, `PastTheGrid`'s direct part, and Auto where it
  picks `direct`); wide v2's (its three rows on the non-double scalars, double's own 64x64x16
  kernel green); `small`'s `Masked` fold (the `small` pins, holes in packed work-groups included);
  the split-k reduce skip (every `splitk` pin); the grid chunk loop dropping the mask slice
  (`PastTheGrid` only). An all-active mask stayed green under every break.
- **W4c, ortho, norm, cond, ritz_values, syevx, iluk** (`tests/norm_mask_tests.cc`,
  `tests/ortho_mask_tests.cc`, `tests/syevx_mask_tests.cc`, `tests/iluk_mask_tests.cc`; acpp Metal
  float and complex float, NETLIB every scalar, 2026-10-28): norm (every type), cond, ortho's CGS2,
  SVQB, SVQB2 and (on a host device) Householder, ritz_values, syevx `Direct`, `LOBPCG` and `Filtered`, and iluk's
  device factorisation and apply run a masked batch; a call whose child still refuses (potrf under
  ortho's Cholesky family, the host BLAS gemm on NETLIB, getri `blocked` under cond on a GPU, ILU(k)
  under LOBPCG) throws before writing. Filtered's batch filter degree is a cross-item host minimum,
  and an ILU(k) pattern is the union over active items: both are checked within a bound. On
  AdaptiveCpp a guard tests a captured bool, not the pointer: a captured null mask read as non-null
  in a Metal kernel, and the JIT's specialisation on the pointer moved CGS2's rounding after a
  projection (ortho(A, M) is checked by value there). Defeating one guard at a time turned red only
  its own cases, on the patterns with an inactive item; ortho's transposed copy is the exception, a
  round trip that rewrites an inactive item with its own bits. syevx `DirectSubset` and lanczos
  carry the mask but refuse until stebz, stein, steqr, sort (W3a) and sytrd_sy2sb honour it.
- **W2b, LU** (`tests/getrf_mask_tests.cc`, `tests/getrs_mask_tests.cc`; Apple M5 Max under
  AdaptiveCpp, Metal float and complex float, NETLIB every scalar, 2026-10-28): getrf `tiny` (I2)
  and `cta` (I1, packed sub-group tiles included), getrs `cta` (I1), gesv `tiny` (I2) and NETLIB's
  getrf, getrs and getri host loops (I4) run a masked batch, as do gesv `blocked` and inv, whose
  children honour it. The `blocked` families of getrf, getrs and getri carry the mask (per-part
  slice, panel leaves, laswp and gather kernels) but refuse until trsm does. Defeating one guard at
  a time turned red only its own pins and the composites over it: tiny's fold (its pins, the
  packed holes, device Auto, and gesv `blocked`); cta's returns (its pins, and gesv `blocked` at
  cta's ceiling); getrs `cta`'s NoTrans return (its NoTrans pins only); gesv `tiny`'s fold (its pins
  only); each host loop's skip (its NETLIB pin and Auto, plus gesv `blocked` over getrf and getrs, inv over
  getri); cta's `info_clear` reverted to a fill (the `info` sentinel only). An all-active mask
  stayed green under every break.
- **W3b and W3c, syev, sygst, sygvd** (`tests/syev_mask_tests.cc`, `tests/sygvd_mask_tests.cc`;
  acpp Metal float and complex float, NETLIB every scalar, 2026-10-28): syev's `cta_fused` and
  `jacobi` and the host backend's LAPACKE syev, sygst and sygvd honour the mask, pinned and
  reached; sytrd_cta, sb2st (both chases and the kd 0 and 1 copies), sb2st_hh with its three
  back-transforms, latrd's legacy and device panels and syev_align are checked unpinned. The
  composites (syev `cta`, `blocked`, `two_stage`; sygst `blocked`, `two_sided`; sygvd `compose`;
  syfn) carry the plumbing and keep refusing until their children land. With each family's guard
  defeated, only the masked patterns that have an inactive item went red (`alternating`, `single`,
  the work-group holes), always as "inactive item written". `none` and `all-active` stayed green,
  and no active item differed. Latrd's grid panel cannot run on Metal or a host device and is
  untested here.
- **W2c, QR** (`tests/geqrf_mask_tests.cc`, `tests/ormqr_mask_tests.cc`; Apple M5 Max Metal float and
  cfloat, NETLIB every scalar, 2026-10-28): geqrf's tiny, cta, blocked and NETLIB families, ormqx_cta,
  NETLIB's ormqr and orgqr, and the blocked ormqr and orgqr wherever the Left update does not reach
  trmm run a masked batch. Defeating one guard at a time turned red only its own cases: tiny's
  `Masked` fold (the tiny shapes); cta's packed sub-group return (cta's packed shapes and blocked's
  short trailing panels, which take that leaf); blocked's per-part slice (its 8-item-part shape);
  ormqx_cta's `live` fold (every ormqx_cta form); orgqr's copy-back skip (the orgqr blocked shapes);
  each NETLIB loop (its op's vendor cases). ormqx_cta's old per-partition return before a barrier
  stayed green on Metal: that fix is held by review, not by a test.
- **W1b, level 3, trsm and spmm** (`tests/trsm_mask_tests.cc`, `tests/level3_mask_tests.cc`,
  `tests/spmm_mask_tests.cc`; sm_89, every scalar, 2026-10-28): every native family of trsm, trmm,
  symm, hemm, syrk, herk, syr2k, her2k and spmm runs a masked batch and every vendor refuses. The
  families over the public gemm (trsm `blocked`, `expand`, `fold`) pin gemm too, and the level-3
  helpers adopt the call's mask onto B and C, since the gemm sees scratch in place of A. spmm's
  gather is bit-identical; its scatter is checked by backward error. Defeating one guard at a time
  turned red only its own cases: trsm `cta`'s I1 return (the `cta` and `blocked` pins, which solve
  their diagonal blocks with `cta`); `sg_left`'s `Masked` fold (every `sg_left` bucket, holes in
  packed sub-groups included); syrk's gram tile (syrk and herk `gram`); `fold_hermitian` (herk and
  her2k `fold`, chunks included); the mirror body (symm and hemm `mirror` except double's tiled
  body); spmm's gather (its NoTrans cases, the scatter green). Dropping the mask slice in
  `level3::batch_range` turned red every family that launches through it, at every masked pattern.
  The unmasked `sg_left` instances keep base's registers and stack on all 44 kernels.
- **W4b, sytrf, sytrs, sysv** (`tests/sytrf_mask_tests.cc`; NETLIB every scalar, Metal float and
  complex float, 2026-10-28): sytrf `cta`, `wg`, `blocked`, sytrs `cta`, `wg`, sysv `cta`, `compose`
  and LAPACKE's sytrf and sytrs host loops run a masked batch; A, the packed ipiv, `info` and B of an
  inactive item keep every byte, a singular item included. `blocked` slices the mask per part, on
  its kernels and its gemm views (the chunk patterns at n = 45 and 170). sytrs `blocked` refuses
  until trsm honours the mask; cuSOLVER's sytrf refuses permanently.

## Rejected alternatives {#masked-batches-rejected}

- **Central compaction** (#207's gather, packed call, scatter). With a device-only mask there is no
  host count, so the children still run at full capacity and nothing is saved; it adds an
  O(k n^2) gather and scatter, and routing by k cost bit-identity on the blocked, two-stage and
  compose families. It stays a possible later optimisation for packed kernels under sparse masks.
- **Templating every kernel on `Masked`.** It doubles device code, and the build is
  device-link-bound. Only the packed families that measured a cost get it.
- **Redirecting vendor pointers to a dummy item.** `info` and pivots are still written, it needs
  staging, and it saves no work. Nor a save-and-restore shield as the vendor default: it saves no
  work and needs full-batch workspace, which gemm has none of.
- **Grid-stride or persistent loops over an active-index list.** A measured dead end: 15% slower at
  saturation.
- **A thread-local or Queue-carried "current mask".** It leaks into unrelated calls and cannot
  express chunk offsets.

# Inbox from shard S2-kernels

No code site points at any of the items below, so `inbox_pointers` is empty. These are line-number citations into files S2 edited that
went stale when the comments shrank (comments only; code unchanged). The current lines are as of 2026-09-30.

## -> docs/design/known-defects.md: stale line citations into the route builders

- Row 7 and the text near line 198 cite `src/backends/trsm_route.hh:51` for
  `s.heterogeneous_batch = A.is_heterogeneous() || B.is_heterogeneous();`. It is now line 40. Citing `trsm_op_shape` by name would stop
  this recurring.
- Near line 62, `gemv_op_shape` (`src/backends/gemv_route.hh:75-78`): the length checks are now lines 39-40.
- New item for the defect ledger if the owner agrees (recorded in `docs/perf/trsm.md` open debt 18): `trsm_op_shape` never sets
  `s.backend`, so every trsm coverage row reads `Backend::AUTO`. `gemv_op_shape` sets it for exactly this reason.

## -> docs/design/vendor-independence.md: stale line citations for the parsed.found spelling

- Lines 113-114 cite `gemv_route.hh:151` and `trsm_route.hh:75` for `parsed.found ? parsed.route : legacy_unset_default(...)`. They are now
  `gemv_route.hh:87` and `trsm_route.hh:66`.

## -> src/util/queue-impl.cc (whoever owns it), line 367

- It cites `src/backends/trsm_route.hh:5-8` for "the layer allowed to query the device". That header comment is now lines 3-5. A pointer
  `evidence: docs/perf/trsm.md#the-shape-builder-and-the-field-mapping` would not drift.

## -> .github/ci/comment_density_waivers.txt (coordinator)

Eight S2 files are now under the 18% ceiling and their waiver lines can be deleted: `src/backends/gemm_heterogeneous.hh` (:125),
`src/backends/gemm_variant.hh` (:126), `src/backends/gemv_route.hh` (:127), `src/backends/trsm_route.hh` (:144),
`src/sycl/device_scalar.hh` (:180), `src/sycl/gemm/register_launchers.hh` (:181), `src/sycl/gemv_native.hh` (:182),
`src/sycl/spmm_native.hh` (:183).

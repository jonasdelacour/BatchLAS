# Inbox from shard R-H6 (review of H6-eigen-api)

## -> docs/design/vendor-independence.md: info spans on syev, gesvd and steqr: forwarder or default

Still owed by the page owner (C1). The section text is in `docs/_inbox/H6-eigen-api.md` under the
same heading and is unchanged, with one correction: the internal six-argument `syev_vendor`
callers after the flat-kernel-selection merge are `src/extra/norm.cc`, `src/extra/cond.cc` and
`src/extensions/syevx_lobpcg.cc`, all through `syev_vendor_or_throw` in `src/ops/syev/vendor.hh`;
`src/backends/cusolverdx.cc` no longer exists. The heading must keep this exact wording (slug
`info-spans-on-syev-gesvd-and-steqr-forwarder-or-default`), because four pointers cite it:

- `include/batchlas/blas/functions/syev.hh`, above the six-argument `syev` forwarder
- `include/batchlas/blas/functions/syev.hh`, above `backend::syev_vendor`
- `include/batchlas/blas/functions/gesvd.hh`, above the two old-arity `gesvd` forwarders
- `include/batchlas/blas/extensions.hh`, the `info` comment block above `steqr`

If C1 would rather not host it (the page is now about flat selection), the natural alternative
home is `docs/design/api-conventions.md` next to `api conventions: per-item info spans`; then the
four pointers above need the new page path in the same change.

## -> docs/cpp-api.md: stale claim in "Convergence status: syev, syevx, gesvd, steqr, stedc"

The paragraph "Two limits to know about" says `stedc`'s status covers its own merges and that a
leaf `steqr` that runs out of sweeps is not reported. That is no longer true: both drivers raise
the caller's `info` for a non-converged leaf (`src/extensions/stedc.cc`, the recursive driver's
leaf goes through `steqr_dispatch`, and the level driver folds a per-leaf status array into the
caller's span, leaf j -> item j / 2^L). `docs/perf/stedc.md#stedc-convergence-reporting-through-info`
already describes the fix. Suggested replacement: "`stein` (inverse iteration, reached through
`syevx`'s `DirectSubset` route) runs a fixed iteration count with no convergence test ..." as the
only remaining limit. The `stedc()` API doc in `extensions.hh` now states the current behaviour.

## -> docs/pages/api_groups.dox: the dispatch group brief

`backend::syev_vendor`, `backend::syev_vendor_buffer_size`, `backend::gesvd_vendor` and
`backend::gesvd_vendor_buffer_size` are `@ingroup dispatch`, whose brief still says
"RouteTable, supports(), preferred()". When C1's `selection` group lands, these four (and the
other `backend::*_vendor` declarations) probably belong there; R-H6 may not use that group id.

## Open items (code, out of scope for a comment-only pass)

- `include/batchlas/internal/sytrd_blocked.hh` declares a `sytrd_blocked` taking `Span<std::byte>`
  by value with no default; `src/extensions/sytrd_blocked.cc` defines only the
  `const Span<std::byte>&` template declared in `extensions.hh`. The internal one is a distinct,
  undefined template (reported by H5; confirmed by R-H6).
- `extensions.hh` declares `sytrd_band_reduction_single_step` and its `_buffer_size` twice each
  (identical signatures); only the second carries the doc block.

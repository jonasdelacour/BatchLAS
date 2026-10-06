# Inbox from shard H5-factor-api

Material moved out of the factorization / QR public headers. H5 owns no pages;
each section below names its target page and the heading the code now cites.

## -> docs/design/vendor-independence.md: Positional validators: reject only what no route can serve

Each positional (workspace-taking) entry point of `geqrf`, `orgqr`, `getrf`,
`getrs`, `getri` and `potrf` has an `<op>_validate_params` in its public header.
It runs in the facade (`src/dispatch/entry_points/factorization.cc`) **ahead of
the shape builder**, because the builder reads `A.rows()` / `A.cols()` and must
not describe a non-conforming view. Same hoist, and same reason, as `trsm`'s
(`entry_points/level3.cc`, the `trsm_validate_params` call before the shape is
built).

**The rule: validate only what no route could serve.** A `supports()` predicate
that says "the native drivers cannot serve this" *routes* the call to the
vendor; it does not say the call is invalid. A validator that threw on such a
shape would turn a currently working positional call into an error, which is a
user-visible behaviour change and belongs in its own commit with its own test,
not in a step whose gate is "zero behaviour change". The option-struct and
arena overloads in `include/batchlas/blas/options.hh` are where the stricter
checks (`require_square`, `require_span_at_least`, `require_info_span`) live.

Per op, what is deliberately *not* checked on the positional path:

* **potrf.** Checks negative extents, squareness and `uplo`. History: there was
  no `potrf_validate_params` anywhere in the tree (the analogous
  `functions/trsm.hh` one existed). `require_square` / `require_info_span` were
  attached only to the option overloads; the workspace-taking `<Backend B>`
  overload -- the spelling `src/extensions/ortho.cc` uses -- had neither, so a
  non-square view reached the backend and cuSOLVER factorised
  `A.rows() x A.rows()` out of it. It does **not** check the length of a
  non-empty `info` span: a short non-empty span silently becomes pool scratch
  (`detail::info_target` in `src/linalg-impl.hh`, whose test is `>= count`),
  documented as by design; turning that into a throw is a behaviour change.
* **geqrf.** One check (negative extents). No squareness check: rectangular A
  is the point of geqrf and the library's own callers pass tall panels
  (`band_reduction.cc`, `sytrd_sy2sb.cc`); copying potrf's `A.rows() != A.cols()`
  there would be a wrong edit. No `m >= n` check, although
  `RouteTable<Op::geqrf,T>::supports()` carries one: that routes a wide view to
  the vendor, which serves it. No `tau` length check: the arena spellings already
  `require_span_at_least`.
* **orgqr.** Negative extents only. No `n <= m` check although `supports()` has
  one: Q's columns live in R^m, so n > m is meaningless, but every backend
  currently accepts such a view and hands it to a vendor; in `supports()` the
  condition merely routes it there. No `tau` length check (arena spellings do it).
* **getrf** (WP6). Negative extents only. No squareness check, although
  `RouteTable<Op::getrf,T>::supports()` and the arena spellings
  (`require_square`) carry one. No pivots length check; the arena spellings do
  `require_span_at_least`.
* **getrs** (WP6). Negative extents only. Squareness of A, `A.rows() == B.rows()`,
  equal batch and the pivot span's length are all checked on the arena
  spellings; the first three make `backend::getrs_op_shape` return `nullopt`,
  which routes the call to the vendor.
* **getri** (WP6). **Two arities, forced by the signatures:** `getri_buffer_size`
  takes A alone while the call takes A and C, and the query must validate
  exactly the view its route is built from -- the route builder is a function of
  A alone (see the header note in `src/backends/getri_route.hh` for why it cannot
  take C). The two arities check A identically; the second adds C's extents,
  which nothing else on the positional path looks at. Neither checks squareness
  of A or C, their agreement in order and batch, or the pivot span's length (all
  checked on the arena spellings); a non-square A makes
  `backend::getri_op_shape` return `nullopt`, which routes to the vendor.

**The contrast: gesv and posv reject more, structurally.** They have no vendor
arm on any backend, so a non-conforming pair would not be served by anyone; it
would reach `throw_no_vendor_route` (or `solve_throw_unroutable`) and report the
wrong cause. Their validators therefore check squareness, `B.rows() == A.rows()`
and equal batch (posv also `uplo`).

Code sites that now point here
(`evidence: docs/design/vendor-independence.md#positional-validators-reject-only-what-no-route-can-serve`):

- `include/batchlas/blas/functions/potrf.hh` (`potrf_validate_params`)
- `include/batchlas/blas/functions/geqrf.hh` (`geqrf_validate_params`)
- `include/batchlas/blas/functions/orgqr.hh` (`orgqr_validate_params`)
- `include/batchlas/blas/functions/getrf.hh` (`getrf_validate_params`)
- `include/batchlas/blas/functions/getrs.hh` (`getrs_validate_params`)
- `include/batchlas/blas/functions/getri.hh` (`getri_validate_params`, one-argument arity)

## -> docs/design/vendor-independence.md: Per-item info spans for potrf, getrf and getri

`info` on `potrf`, `getrf` and `getri` (and, through them, `posv` and `gesv`) is
the LAPACK per-item status: one int32 per batch item, 0 on success, and > 0 for
the leading minor that is not positive definite (potrf), the column at which U
became exactly singular (getrf), or the zero diagonal of U that leaves the item
without an inverse (getri).

History (issue #73): it used to be unreachable. Every backend allocated the array
the vendor call needs, passed it, and dropped it, so a caller could not tell a
batch that factorised from one where item 37 is rank-deficient (potrf), singular
with every downstream `getrs`/`getri` producing noise (getrf), or left holding a
matrix of infinities (getri).

Design points that are part of the contract:

* An **empty** span means "not requested" and is exactly the previous behaviour:
  the backend falls back to its own scratch allocation.
* The **workspace size is deliberately the same either way**, so
  `*_buffer_size` stays correct whether or not a caller asks for status.
* A **short non-empty** span is silently treated as "not requested" on the
  positional path (`detail::info_target`, `src/linalg-impl.hh`); the option
  overloads reject it with `require_info_span`. See the potrf item under
  "Positional validators" for why the positional path does not throw.
* **Old-arity forwarders, not a defaulted parameter.** `info` cannot be a
  defaulted trailing parameter: the `sig::` aliases used for explicit
  instantiation (`BATCHLAS_INSTANTIATE`, `src/util/template-instantiations.hh`)
  are function *types*, and function types cannot carry default arguments, so a
  default would not be part of the instantiated signature. A separate inline
  overload keeps every existing call site of the old arity compiling unchanged.
  `gesv` and `posv` follow the same pattern.

Code sites that now point here
(`evidence: docs/design/vendor-independence.md#per-item-info-spans-for-potrf-getrf-and-getri`):

- `include/batchlas/blas/functions/potrf.hh` (`potrf` declaration)
- `include/batchlas/blas/functions/getrf.hh` (`getrf` declaration)
- `include/batchlas/blas/functions/getri.hh` (`getri` declaration)

## -> docs/perf/qr.md: ormqr: one route resolution for the call and its size query

`ormqr_route` (in `include/batchlas/blas/functions/ormqr.hh`) is the single
resolution shared by `ormqr_dispatch` and `ormqr_buffer_size_dispatch`, built
from the same pure inputs (`ormqr_op_shape`, the parsed `BATCHLAS_ORMQR_ROUTE`,
and `resolve_ormqr_block_size`, which reads A alone).

It replaced `choose_ormqr_provider`, which returned a forced provider without
checking it against `ormqr_supports_blocked`; `route_ormqr.hh`'s header records
the two defects that followed. The one that concerns this header: a forced
provider that was neither Vendor nor Blocked reached `ormqr_dispatch`'s `else`
arm (`else { chosen = Vendor; ... }`) and ran on the vendor, while
`ormqr_buffer_size_dispatch` fell past its single `if` and returned the
**blocked** size. Sizing a workspace with the public query and passing it to the
public call therefore threw `ormqr: insufficient workspace for chosen provider`
(the 108x case recorded in `docs/perf/dispatch.md`, Correctness findings: 2560
bytes against the 276480 the call demanded, on every GPU type). With the pure
resolver there is no third arm: the result is either a vendor route or a
supported native one.

Also recorded here: the unset default for ormqr is Auto, unlike GEMM's Vendor.
The `block_size_hint` exists because the tuning table
(`tuning::ormqr_block_size_for_n`) is keyed on `A.rows()`, the panel height,
which for a tall skinny panel is the wrong dimension; a caller that knows the
reflector count k can pick the width, and the hint is clamped to k so it never
exceeds the number of reflectors.

Code sites that now point here
(`evidence: docs/perf/qr.md#ormqr-one-route-resolution-for-the-call-and-its-size-query`):

- `include/batchlas/blas/functions/ormqr.hh` (`detail::ormqr_route`)
- `include/batchlas/blas/functions/ormqr.hh` (`ormqr_buffer_size_dispatch`)

## Notes for the coordinator

- The six `// The vendor path for <op>. DECLARATION ONLY ... WP0 S5 moves that
  definition to src/dispatch/entry_points/factorization.cc` blocks (geqrf,
  getrf, getri, getrs, orgqr, potrf) were cut to one line plus
  `evidence: docs/design/vendor-independence.md#the-entry-point-facade`; that
  section already records the move, so no text needed to be added.
- `ormqr_vendor_or_throw` now points at the existing
  `docs/design/vendor-independence.md#the-vendor-gate`.
- Waivers now unneeded (all eight files are under 18%): see the shard result.
- For the owner of `include/batchlas/blas/extensions.hh`: `ormbr`, `ormbr_buffer_size`
  and `sytrd_blocked_buffer_size` are declared twice (there, with default
  arguments, and in `include/batchlas/internal/`). A doc block on the internal
  redeclaration makes Doxygen warn "no matching file member found", so the
  internal headers now document them in their `@file` block only. The ormbr
  contract written there (vect Q/P orders, tau unit-stride and packed by batch,
  `unsupported` for complex `Trans` with `'P'`) could be copied onto the
  extensions.hh declaration.
- Possible API defect, not a comment issue: `internal/sytrd_blocked.hh` declares
  `sytrd_blocked(..., Span<std::byte> ws, int32_t block_size)` (by value, no
  default) while `blas/extensions.hh` declares
  `sytrd_blocked(..., const Span<std::byte>& ws, int32_t block_size = ...)`.
  These are two distinct function templates, not a redeclaration; worth checking
  which one `src/` defines and instantiates.

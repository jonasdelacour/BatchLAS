# Inbox from shard H6-eigen-api

Material cut from `include/batchlas/blas/extensions.hh`, `include/batchlas/blas/functions/syev.hh`
and `include/batchlas/blas/functions/gesvd.hh`. The user-facing contract of `info` is already in
`docs/cpp-api.md#convergence-status-syev-syevx-gesvd-steqr-stedc` and was not duplicated; what
follows is the declaration-shape rationale and the history that the code comments carried.

## -> docs/design/vendor-independence.md: info spans on syev, gesvd and steqr: forwarder or default

Five routines (`syev`, `syevx`, `gesvd`, `steqr`, `stedc`) and every tier below them take a
trailing `Span<int32_t> info`, and two different declaration shapes carry it. Both keep every
pre-existing call site compiling unchanged; which one an entry point uses is forced by how it is
explicitly instantiated.

**Public `syev` and `gesvd`: an old-arity forwarder, not a defaulted parameter** (the same
shape as `potrf.hh`). What forces it is the `sig::syev` / `sig::gesvd_vendor` alias used by
`BATCHLAS_INSTANTIATE` (`src/util/template-instantiations.hh`): it is a function *type*, and
function types cannot carry default arguments, so `info` has to be spelled out in the alias
whichever way the declaration is written. Leaving the declaration default-free too keeps alias
and declaration parameter-for-parameter identical, which is the invariant `BATCHLAS_INSTANTIATE`
reads. The inline forwarder is then what keeps the six-argument `syev` and the eight- and
nine-argument `gesvd` call sites (the `SyevOptions` / `GesvdOptions` spellings in
`blas/options.hh` among them) compiling. For `gesvd`, arity plus the `Uplo`/`Span` type
difference at parameter 8 keeps all four overloads unambiguous.

**`backend::syev_vendor` and `backend::gesvd_vendor`: a defaulted `info_out`.** A default
argument is a property of the declaration, not of the function type, so `sig::syev_vendor`
still names the full seven-parameter signature (nine for `gesvd_vendor`) and the explicit
instantiations in the vendor TUs still match. That default is what keeps the internal
six-argument `syev_vendor` callers (`src/extra/norm.cc`, `src/extra/cond.cc`,
`src/extensions/syevx_lobpcg.cc`, `src/backends/cusolverdx.cc`) compiling with no extra
overload; none of them is public API, so none needs a forwarder.

**`steqr`, `steqr_cta`, `stedc`, `syevx*`, the `syev_*` tiers and the SVD tiers
(`extensions.hh`): a defaulted trailing parameter.** They are instantiated from hand-written
macros that spell the parameter list out, not from a `sig::` alias, and they already default
`jobz`, `params` and `eigvects`, so an old-arity forwarder would be *ambiguous* with the primary
rather than additive.

History, from the comments this replaces: before the `info` span existed, every vendor already
allocated a per-item status array because the vendor call demands somewhere to write, and it
was pool scratch that nobody read. cuSOLVER's `gesvdjBatched` returned an info array that the
library dropped; the netlib path captured LAPACKE's scalar `info` per item and threw it away in
a batch-wide exception. `backend::gesvd_vendor` also used to be *defined* in
`functions/gesvd.hh` (a NETLIB LAPACKE loop plus a throw for every other backend), which made a
CUDA definition in `src/backends/cusolver.cc` a redefinition error rather than an override --
the reason there was never a cuSOLVER SVD binding until the header became declaration-only (the
LAPACKE body now lives in `src/backends/netlib_lapack.cc`; see
`docs/design/gesvd.md#gesvd-design-vendor-binding-and-dispatch`).

Code sites that now point here
(`evidence: docs/design/vendor-independence.md#info-spans-on-syev-gesvd-and-steqr-forwarder-or-default`):

- `include/batchlas/blas/functions/syev.hh`, comment above the six-argument `syev` forwarder
- `include/batchlas/blas/functions/syev.hh`, comment above `backend::syev_vendor`
- `include/batchlas/blas/functions/gesvd.hh`, comment above the two old-arity `gesvd` forwarders
- `include/batchlas/blas/extensions.hh`, the `info` comment block above `steqr`

## Open items found while documenting (not fixed: code changes are out of scope)

- `francis_sweep` (`extensions.hh`) is declared `BATCHLAS_API` with no definition or explicit
  instantiation anywhere in `src/`: a call compiles and fails at link time. The API doc now says
  so with `@warning`. Candidate for `docs/design/known-defects.md`, or delete the declaration.
- `OrmqCtaFactorization` (`extensions.hh`) is referenced by no entry point, test or source file.
- `tridiagonal_solver` addresses `Q` with stride `n` in its rotation update (`Q[k*m+l]` in
  `apply_all_reflections`) while the identity fill uses `Q.ld()`; a `Q` with `ld() != n` gets a
  wrong answer. Documented as `@pre Q.ld() == n`. It caps QR steps at six per eigenvalue with no
  convergence report, and nothing in `src/` calls it.
- `sytrd_band_reduction_single_step` is declared twice in `extensions.hh` (identical
  signatures); harmless but one declaration can go.

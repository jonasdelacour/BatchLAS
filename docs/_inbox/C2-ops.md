# C2-ops handoff

## -> docs/developer/documentation.md: Documenting an op's choices

Proposed as a new `###` subsection under "Writing API documentation" (heading text above is
site-unique; slug `documenting-an-ops-choices`). Every `src/ops/<op>/choice.hh` follows it, so a
new op (PR 147's symm, syrk, syr2k, trmm, and any later one) copies the pattern and needs no page
edits: the `selection_ops` group lists whatever files add themselves to it.

Text:

> `src/ops/<op>/choice.hh` is the op's selection vocabulary (see @ref selection). It lives in
> `src/`, so its doc comments count toward the 18% density ceiling; the pattern below fits in
> about four comment lines because it puts almost everything in trailing `///<` comments, which
> the counter scores as code.
>
> 1. **File block, two lines.** `/// @file`, then
>    `/// @brief <op>: <family>, <family>, .... evidence: docs/perf/<page>.md @ingroup selection_ops`.
>    The brief names the families in candidate order (the group page lists it), the evidence
>    pointer names the op's `docs/perf/` page, and `@ingroup selection_ops` puts the file in the
>    group. Keep the line within 120 columns; drop the anchor before dropping the group.
> 2. **Each field-less family**: one trailing `///<` on its `struct X : select::NoFields<"x"> {};`
>    line: the driver it runs, then its `can_run` limits beyond the op's common native term
>    (`n <= 32`, `Lower only`, `needs the solver library`).
> 3. **A family with fields** (gemm's `reg`/`wide`, potrf's `lpanel`): a preceding `///` block (a
>    trailing `///<` on `struct X {` documents the first member, not the struct) giving the driver,
>    what each field means and which values are legal. A field declared alone gets its own `///<`;
>    on a multi-name declaration (`int m, n, k;`) only the last name would get it, so describe them
>    in the struct block.
> 4. **The choice variant**: trailing `///<` with the common native term (`GPU, sub-group 32,
>    square, no NETLIB`) or, for an op with no vendor, that a `vendor` pin warns and runs Auto.
> 5. **`candidates<T>()`**: `template <class T>  /// Every compiled choice, in tie-break order (§6.3).`
>    A `///` after code on the `template` line attaches to the function and counts as code.
> 6. **`spec`, `last_resort`, `key_names`, `grid_*`**: trailing `///<` (on the first or the last
>    line of a multi-line initialiser; both attach). Key comments say what each key measures and
>    why its `:log` weight is what it is (work ~ n^3 means weight 3); grid comments say which old
>    thresholds the grid straddles and that the transcriber (`tuned/README.md`) spells the same grid.
> 7. **Section numbers** `§x.y` and rule ids `R1`-`R8` refer to `docs/design/flat-kernel-selection.md`.
>    Cite a section by number with a page-only pointer, `evidence: docs/design/flat-kernel-selection.md (§5.4)`:
>    its numbered headings get `autotoc_md` ids on the site, so a `#54-...` anchor is not live. Cite any other anchor in a
>    plain comment, or wrapped in backticks inside a `///` comment: Doxygen reads `#anchor` in a
>    doc comment as an explicit link request and warns, while the checker's pattern stops at the
>    backtick, so a backticked pointer is still checked.
> 8. **Helpers next to the vocabulary** (`can_run.hh`, `vendor.hh`, sizing helpers such as
>    `geqrf.hh`) get the same two-line file block or a function doc ending in
>    `@ingroup selection_ops`, and are listed in the Doxyfile `INPUT` next to `choice.hh`.
> 9. **`<op>.cc` is not on the site** (`FILE_PATTERNS` has no `*.cc`). Its comments stay plain:
>    a 2-4 line header (the op, R1, which family runs which driver), `// Correctness only (R3)`
>    above `can_run` naming what each clause guards, `// Exactly the chosen family's need (R5)`
>    above the workspace visitor, and invariants at the line they guard. Measurements and history
>    go to the op's `docs/perf/` page.

Code sites: every `src/ops/*/choice.hh` in this branch (gemm, gemv, geqrf, gesv, gesvd, getrf,
getri, getrs, orgqr, ormqr, posv, potrf, spmm, syev, trsm), plus `src/ops/geqrf/can_run.hh`,
`src/ops/geqrf/geqrf.hh`, `src/ops/ormqr/vendor.hh`, `src/ops/syev/vendor.hh`. No code pointer
cites this heading yet.

## -> docs/perf/qr.md: (no new heading) append under "The `orgqr_buffer_size` latent defect"

Text to append as a final paragraph of that section:

> **The vendor's orgqr workspace is batch-linear.** The vendor family is a per-item loop
> (`backend::orgqr_vendor`), so its workspace grows with the batch; a native call must never be
> sized by it. At cdouble n=64 batch=8192 the vendor size is ~4.6 GB. (Moved from the
> `src/ops/orgqr/orgqr.cc` workspace comment, where it was stated without a measurement record;
> it is an arithmetic size, not a timing.)

Code site: `src/ops/orgqr/orgqr.cc` (workspace visitor comment) keeps
`evidence: docs/perf/qr.md#the-orgqr_buffer_size-latent-defect`, which already resolves.

## Notes for the coordinator

- `src/ops/gemm/gemm.cc` `can_run` comment: the history "before this, 18 NN-only variants silently
  computed NN on a transposed call" was removed from code; `docs/perf/gemm.md` already records it
  ("Correctness, new in P3.4", under `### Choices (flat selection, P3.4)`), and the code now points
  at `docs/perf/gemm.md#choices-flat-selection-p34`. No page edit needed.
- `src/ops/getri/getri.cc`: the dangling pointer `docs/perf/lu.md#correctness-findings` was
  repointed to `docs/perf/lu.md#lu-correctness-findings`. `tests/getri_candidates_tests.cc:547`
  carries the same dangling pointer; whoever owns `tests/` should make the same one-word fix.
- Doxyfile `INPUT` additions requested (headers only): `src/ops/gemm/choice.hh`,
  `src/ops/gemv/choice.hh`, `src/ops/geqrf/choice.hh`, `src/ops/geqrf/can_run.hh`,
  `src/ops/geqrf/geqrf.hh`, `src/ops/gesv/choice.hh`, `src/ops/gesvd/choice.hh`,
  `src/ops/getrf/choice.hh`, `src/ops/getri/choice.hh`, `src/ops/getrs/choice.hh`,
  `src/ops/orgqr/choice.hh`, `src/ops/ormqr/choice.hh`, `src/ops/ormqr/vendor.hh`,
  `src/ops/posv/choice.hh`, `src/ops/potrf/choice.hh`, `src/ops/spmm/choice.hh`,
  `src/ops/syev/choice.hh`, `src/ops/syev/vendor.hh`, `src/ops/trsm/choice.hh`. A glob-free
  alternative that also picks up PR 147's ops with no further edit: add `src/ops` to `INPUT`
  (`FILE_PATTERNS` does not match `.cc`, so the op files stay off the site); the only `.hh` files under `src/ops`
  are the ones listed (after PR 147 also `src/ops/level3/*.hh` and the four new `choice.hh`, which
  render without warnings only once documented in this pattern).
- No earlier inbox item targets a file this shard owns (`src/ops/**`), so nothing was applied.

## -> docs/design/flat-kernel-selection.md: Which inputs the old routers read, per op

Added by the C2-ops review. Key-provenance history removed from `choice.hh` comments (it explains
why each op's `key_names` are what they are, but is history, not an invariant). Suggested text:

> The table keys were chosen to cover every input the pre-flat routers' predicates read, plus the
> work estimate. getri: the old router read only n (batch >= 1 was a correctness term, now in
> `can_run`). gesv: the order and nrhs (batch only as >= 1). orgqr: m and n only, no batch and no
> architecture, so its table has no batch key.

Code sites: `src/ops/getri/choice.hh`, `src/ops/gesv/choice.hh`, `src/ops/orgqr/choice.hh`
(`key_names`). No code pointer cites this heading.

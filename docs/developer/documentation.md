# Documentation conventions {#documentation_conventions}

> **Status:** current · governs the whole `docs/` tree.

How the site is built, where material lives, the anchor contract between code comments and pages,
and how to write API documentation and record results. The code points at the site for every
routing window, rejected kernel and measured ratio it relies on.

## Style

- Reference-book style: what a thing is, how to use it, the numbers that matter. Open with the H1,
  one status line, then a one-to-three sentence summary.
- Short paragraphs, bullets and tables, present tense, active voice, no "we", no narrative history.
  A deleted feature someone may look for gets one sentence.
- One sentence of "why" only where a reader needs it to avoid breaking a threshold, invariant or
  trap. Say each fact once and link to it.
- Keep only the numbers that back a shipped decision (threshold, window, constant, design choice):
  the winner and the bracketing loser on each edge, with machine, date and raw-data path. Git
  history keeps the rest. Rejected alternatives: one `| Alternative | Result | Verdict |` row each.
- A cited heading stays (it may shrink to a sentence) or gets its old slug pinned; see
  [Anchors are the contract](#anchors-are-the-contract). Delete stale claims you cannot confirm.
- `> **Note:**` / `> **Warning:**` blockquotes for genuine traps only.

## Building the site

```sh
sh scripts/build_docs.sh                  # -> build/docs/html/index.html
DOXYGEN=/path/to/doxygen sh scripts/build_docs.sh /tmp/site
BATCHLAS_DOCS_STRICT=1 sh scripts/build_docs.sh   # warnings are errors, as in CI
```

`scripts/build_docs.sh` is the only build path: it generates the database pages
(`docs/tools/gen_db_pages.py`), runs `doxygen docs/Doxyfile` from the repository root, then
`docs/tools/gen_site_assets.py` (navigation tree, search index, cache-busting `?v=` hashes) and
`docs/tools/check_doc_anchors.py` against the XML. Warnings go to `<out>/doxygen-warnings.log`.
No SYCL tree is needed.

| Entry point | What it does |
| --- | --- |
| `-DBATCHLAS_BUILD_DOCS=ON`, target `batchlas_docs` | `cmake/BatchLASDocs.cmake` calls the script into `<build>/docs`; `cmake --install` installs `html/`. |
| `sh .github/ci/run_local_checks.sh` | `gen_db_pages.py --self-test`, then the build when Doxygen 1.18+ is on `PATH` or in `DOXYGEN`. |
| CI job `docs` | Pinned Doxygen, `BATCHLAS_DOCS_STRICT=1`, uploads `docs-html`; `docs-deploy` publishes to Pages on push to `main` ([CI](../ci.md)). |

- **Doxygen is pinned to 1.18.0** (`DOXYGEN_VERSION`/`DOXYGEN_SHA256` in `.github/workflows/ci.yml`);
  the script refuses older versions. To bump: change both, build with `BATCHLAS_DOCS_STRICT=1`,
  re-run `check_doc_anchors.py --xml <out>/xml` (heading-id rules change between versions), and
  regenerate `docs/theme/header.html` with `doxygen -w html`, re-applying the `bl-*.js` and
  `doxygen-awesome-*.js` script lines and the `#titlearea` markup.
- **Theme:** doxygen-awesome-css 2.5.0 vendored in `docs/theme/doxygen-awesome/` (its `LICENSE`
  stays beside it), plus `docs/theme/batchlas.css`, `bl-site.js`, `DoxygenLayout.xml`. Doxygen's
  tree view, outline panel and search are off.
- A page appears in the navigation only if some section page lists it with `@subpage`.

## Where things go

| Material | Location | Label prefix |
| --- | --- | --- |
| Measurements, A/B results, rejected alternatives, bracketing evidence for a routing window | `docs/perf/<op>.md` | `perf_<op>` |
| Design decisions, API design records, known defects | `docs/design/<topic>.md` | `design_<topic>` |
| Mathematics and derivations | `docs/algorithms/<topic>.md` | `algo_<topic>` |
| User guides | `docs/guide/<topic>.md` | `guide_<topic>` |
| Process and tooling | `docs/developer/<topic>.md` | `dev_<topic>` |
| Raw result files (CSV, logs) | `benchmarks/results/` in Git LFS | catalogued by the Results database page |
| API contract | Doxygen comments in `include/batchlas/` | `@ingroup` |
| Invariants, deliberate oddities, traps | a short code comment plus an evidence pointer | none |

- **Markdown lives under `docs/`.** Exceptions: a directory `README.md` (never at the root),
  `.claude/` (`.claude/CLAUDE.md` only imports the [agent core rules](agent-core.md)) and `.github/`
  (the project README is `.github/README.md`). `.github/ci/check_markdown_locations.py` enforces it.
  No plan files: write conclusions into the area's page, keep task lists out of the tree.
- **A code comment carries only** an invariant, a deliberate oddity or a trap, plus
  `evidence: docs/<path>.md#<anchor>`; the 18% per-file ceiling (`check_comment_density.py`) keeps
  lab notes out of the source.
- **Math:** `\f$ ... \f$` inline, `\f[ ... \f]` display; `$...$` is not rendered.
- **Links:** relative Markdown links or `@ref <label>`. A broken link is a warning; CI fails on warnings.

## Anchors are the contract

Hundreds of comments in `src/`, `include/`, `tests/`, `benchmarks/` and CMake cite
`evidence: docs/<page>.md#<anchor>`, where the anchor is the GitHub slug of the heading, so a
heading's wording is an interface.

- **Shortening a cited heading:** pin the old slug, `## Shorter title {#exact-old-slug}`. It may
  shrink to a sentence, but not be deleted while cited. Update every pointer in the same change.
- **Distinctive text:** `.github/ci/check_evidence_anchors.py` resolves every pointer and requires
  each cited slug to be unique across `docs/` (Doxygen numbers repeats site-wide). Write
  `syevx: the bisection crossover`, not `Results`.
- **No leading numbers:** Doxygen gives `## 11. Foo` an `autotoc_md` id, so its slug is not live on
  the site (known-defects uses `## Defect 11: ...`).
- `check_doc_anchors.py` (run by `build_docs.sh`) checks each cited anchor in Doxygen's XML. Before
  rewording or deleting a section, read the generated Evidence index (`@ref evidence_index`).

## Writing API documentation

Public headers (`include/batchlas/`, `include/batchlas.hh`) state the **contract**: what the
function computes, each parameter, the return value, preconditions, what it throws, and workspace
needs. Rationale goes to a design or evidence page, linked with `@see` or an evidence pointer.

```cpp
/// @brief One-line summary ending with a full stop.
/// @param queue  queue the kernels are enqueued on
/// @param a      batch of n x n matrices; overwritten with the factor
/// @return       event of the last enqueued kernel
/// @pre  a.rows() == a.cols()
/// @throws batchlas::Error if the workspace is too small
/// @ingroup api_factorizations
```

- `JAVADOC_AUTOBRIEF` is on (first sentence is the brief); `///<` for trailing member comments.
  Aliases: `@invariant`, `@trap`, `@evidence{label}`.
- Every public entity belongs to a group from `docs/pages/api_groups.dox` (all under
  `%api_reference`, each optionally with a `%<group>_lowlevel` child; `%api_selection` /
  `%api_selection_ops` cover `src/select/` and `src/ops/`). Use `@ingroup <id>` or
  `@addtogroup <id>` ... `@{` ... `@}`; add new groups in `api_groups.dox`, not a header.
- **Comment-density exemption:** in the public headers only, doc-comment lines (`///`, `//!`,
  `/**`, `/*!` blocks) count in neither side of the 18% ceiling (`check_comment_density.py --all`
  reports them as `doc`). `/***` banners, `////` rules and mixed lines count. Rationale written
  with `///` is still rationale: move it to `docs/`.

### Documenting an op's choices

`src/ops/<op>/choice.hh` (see @ref api_selection, @ref design_flat_selection) is on the site but in
`src/`, so its doc comments count toward the 18% ceiling. Trailing `///<` comments score as code,
so this pattern documents a vocabulary in about four comment lines; a new op copies it with no
page edits.

1. **File block:** `/// @file`, then `/// @brief <op>: <family>, .... evidence: docs/perf/<page>.md @ingroup %api_selection_ops`
   (families in candidate order, within 120 columns; drop the anchor before the group).
2. **Field-less family:** one trailing `///<` on `struct X : select::NoFields<"x"> {};`: the
   driver, then `can_run` limits beyond the op's common native term.
3. **Family with fields:** a preceding `///` block (driver, each field and its legal values). A
   trailing `///<` on `struct X {` documents the first member; on `int m, n, k;` only the last name.
4. **Choice variant:** a trailing `///<` with the common native term, or that a `vendor` pin warns
   and runs Auto when the op has no vendor family.
5. **`candidates<T>()`:** `template <typename T>  /// Every compiled choice, in tie-break order (§6.3).`
6. **`spec`, `last_resort`, `key_names`, `grid_*`:** a trailing `///<` or a short `///` block; keys
   say what they measure and why the `:log` weight is what it is, grids which thresholds they straddle.
7. **`§x.y` and `R1`-`R8`** refer to `docs/design/flat-kernel-selection.md`; cite by number with a
   page-only pointer, `evidence: docs/design/flat-kernel-selection.md (§5.4)`. Backtick any other
   `#anchor` inside a `///` comment (Doxygen reads it as a link; the checker still sees it).
8. **Helpers** (`can_run.hh`, `vendor.hh`, sizing headers) get the same file block or a function
   doc ending in `@ingroup %api_selection_ops`.
9. **`<op>.cc` is not on the site:** a 2-4 line header, `// Correctness only (R3)` above
   `can_run`, `// Exactly the chosen family's need (R5)` above the workspace visitor, invariants at
   the line they guard.

## Adding results

1. **Raw data** goes in `benchmarks/results/` (Git LFS; `git lfs install` once per machine, or
   `.github/ci/check_lfs_pointers.py` fails CI). Name it `<campaign>_<op>_<dtype>[_<machine>].csv`.
   Machine tokens are `rtx4090` and `sm120` (threadripper02); add new ones to `MACHINES` in
   `docs/tools/gen_db_pages.py`, or the row claims the primary box.
2. **The evidence section** records, next to the numbers, the machine, the date, the command with
   its flags (batch, warm-up, iterations, pinned route) and the raw file path. See @ref perf_evidence.
3. **The Results database** (`@ref results_database`) is generated from `benchmarks/results/` on
   every build; `—` in its pages column means no page distils the file yet.
4. Older raw evidence is at the tag `perf-evidence/vendor-independence`
   (`git show perf-evidence/vendor-independence:experiments/<path>`).

## Page template

````md
# <op or topic>: <what the page is about> {#perf_<op>}

> **Status:** current · RTX 4090 (sm_89), CUDA 13.2 · measured <dates>

<One to three sentences: the conclusion needed first.>

## <op>: the shipped predicate

<Quote the predicate with file:line.>

## <op>: <distinctive name of the window or decision>

<Winner and bracketing loser on each edge. Raw data: benchmarks/results/<file>.csv. Command: <exact command>.>

## <op>: rejected alternatives

| Alternative | Result | Verdict |
| --- | --- | --- |

## <op>: open debts
````

Status line: `current` or `historical, superseded by @ref <label>.`, then machine (measurement pages
only) and date, separated by `·`. Other page kinds use their prefix from
[Where things go](#where-things-go) and drop the machine.

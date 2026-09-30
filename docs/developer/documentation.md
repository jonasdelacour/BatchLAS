# Documentation conventions {#documentation_conventions}

> **Covers:** how the documentation site is built, where each kind of material
> lives, the anchor contract between code comments and pages, how to write API
> documentation, and how to record new results.
> **Status:** current; this page governs the whole `docs/` tree. Introduced
> 2026-09-30 with the Doxygen site.

The site is more than API reference. It is the project's **database of
results, rationale and design decisions**: every routing window, rejected
kernel and measured ratio that the code relies on is written down here, and the
code points at it. The rules below keep that database consistent: one place
for each kind of fact, links that are checked, and API docs that describe the
contract.

## Building the site

```sh
sh scripts/build_docs.sh                  # -> build/docs/html/index.html
DOXYGEN=/path/to/doxygen sh scripts/build_docs.sh /tmp/site
BATCHLAS_DOCS_STRICT=1 sh scripts/build_docs.sh   # warnings are errors, as in CI
```

`scripts/build_docs.sh` is the only build path. It checks the Doxygen version,
generates the database pages (`docs/tools/gen_db_pages.py`, see
[Adding results](#adding-results)), runs `doxygen docs/Doxyfile` from the
repository root, and then runs `docs/tools/check_doc_anchors.py` against the XML
output. Warnings go to `<out>/doxygen-warnings.log`; the script prints how many
lines it holds. Nothing in it needs a configured SYCL tree or a compiler.

| Entry point | What it does |
| --- | --- |
| `sh scripts/build_docs.sh [out]` | The build. Default out dir `build/docs`. |
| `-DBATCHLAS_BUILD_DOCS=ON`, target `batchlas_docs` | CMake wrapper (`cmake/BatchLASDocs.cmake`) that calls the script with `DOXYGEN` set to the doxygen CMake found, into `<build>/docs`. `cmake --install` installs `html/` under the doc dir when it has been built. |
| `sh .github/ci/run_local_checks.sh` | Runs `gen_db_pages.py --self-test`, then the build into `build/docs` when Doxygen 1.18+ is on `PATH` (or in `DOXYGEN`); prints a skip line otherwise. |
| CI job `docs` | Hosted. Installs the pinned Doxygen, runs the script with `BATCHLAS_DOCS_STRICT=1`, uploads `build/docs/html` as the `docs-html` artifact. `docs-deploy` publishes it to GitHub Pages on a push to `main`. See [the CI page](../ci.md). |

**Doxygen is pinned to 1.18.0.** The script refuses anything older, and CI
installs the official release tarball and checks its sha256 against the value
in `ci.yml`. The reasons for the pin: the Doxyfile uses
`MARKDOWN_ID_STYLE = GITHUB`, and the site-wide heading-id behaviour that the
anchor contract depends on (below) was verified on 1.18; the vendored theme's
`docs/theme/header.html` is the 1.18 default header with the theme's scripts
added. Distribution packages (Ubuntu 22.04 ships 1.9.1) are too old.

**Theme.** [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css)
2.5.0 is vendored under `docs/theme/doxygen-awesome/` together with its MIT
`LICENSE`, which must stay beside it. `docs/theme/batchlas.css` holds our
colour tokens on top of it; layout is the theme's. `docs/theme/DoxygenLayout.xml`
sets the navigation.

**Bumping either.** For Doxygen: change `DOXYGEN_VERSION` and `DOXYGEN_SHA256`
in `.github/workflows/ci.yml` (hash the release's `linux.bin.tar.gz` yourself),
build locally with `BATCHLAS_DOCS_STRICT=1`, and re-run
`python3 docs/tools/check_doc_anchors.py --xml <out>/xml`, because heading-id
rules are what change between versions. Regenerate the header with
`doxygen -w html header.html footer.html style.css` and re-apply the five
`doxygen-awesome-*.js` script lines. For the theme: replace the files in
`docs/theme/doxygen-awesome/` (including `LICENSE`) from the release, update the
version in the first line of `docs/theme/header.html`, and compare the theme's
own recommended header against ours.

## Where things go

| Material | Location | Label prefix |
| --- | --- | --- |
| Measurements, grids, A/B results, rejected alternatives, bracketing evidence for a routing window | `docs/perf/<op>.md` (evidence pages) | `perf_<op>` |
| Design decisions, API design records, "why this architecture", known defects | `docs/design/<topic>.md` | `design_<topic>` |
| Mathematics and derivations | `docs/algorithms/<topic>.md` | `algo_<topic>` |
| User guides | `docs/guide/<topic>.md` | `guide_<topic>` |
| Process and tooling (CI, this page, profiling workflows) | `docs/developer/<topic>.md` | `dev_<topic>` |
| Raw result files (CSV, logs) | `benchmarks/results/` in Git LFS | none; catalogued by the Results database page |
| API contract (what a function does, its parameters, preconditions) | Doxygen comments in `include/batchlas/` | `@ingroup` |
| Invariants, deliberate oddities, traps | a short code comment plus an evidence pointer | none |

Rules that go with the table:

- **Markdown lives under `docs/`.** Only `README.md` (any directory), the root
  `AGENTS.md`, `CLAUDE.md` and `LICENSE.md`, and files under `.github/` may live
  elsewhere. `.github/ci/check_markdown_locations.py` enforces this in CI. A
  plan file at the repository root is the thing this rule exists to stop: write
  the plan's *conclusions* into the page for its area, and keep task lists out
  of the tree.
- **A code comment carries only** an invariant, a "looks wrong but is
  deliberate" note, or a trap, plus a pointer of the form
  `evidence: docs/<path>.md#<anchor>` to where the reasoning and numbers live.
  The per-file comment-density ceiling (18%, `.github/ci/check_comment_density.py`)
  is what keeps lab-notebook material out of the source.
- **A new page** starts with `# Title {#label}`, the label snake_case with the
  prefix from the table, and a short status block under it; see
  [Page template](#page-template).
- **Math** uses Doxygen's MathJax syntax: `\f$ ... \f$` inline and
  `\f[ ... \f]` for display. GitHub-style `$...$` is not rendered by the site.
- **Links between pages**: relative Markdown links, such as
  `[the LU page](../perf/lu.md#<anchor>)`, or `@ref <label>`. Both are checked
  by the build: a broken one is a warning, and CI fails on warnings.
- **Preserve information when you migrate or distil.** Keep every measured
  number, the conclusions, rejected alternatives with their reason, open debts
  and traps. Drop checklists and superseded speculation, but say what
  superseded what. Mark a partly stale claim as stale rather than deleting it.

## Anchors are the contract

Several hundred comments (447 pointers on 2026-09-30) in `src/`, `include/`, `tests/`, `benchmarks/` and the CMake
files cite a heading with `evidence: docs/<page>.md#<anchor>`. The anchor is the
GitHub slug of the heading text (lower case, punctuation dropped, spaces to
hyphens). That makes a heading's wording an interface: renaming it breaks every
pointer to it.

Two checks guard it:

- `.github/ci/check_evidence_anchors.py` (CI job `comment-density`) resolves
  every pointer in the tree against the headings of the named page, and also
  requires every **cited** slug to be unique across all of `docs/`. Doxygen
  numbers a repeated heading site-wide (the second `Results` anywhere on the
  site becomes `results-1`), while GitHub numbers per page, so a cited heading
  that is unique on its page but not on the site would work on GitHub and
  break on the site. Give any heading that code will cite distinctive text:
  `syevx: the bisection crossover`, not `Results`.
- `docs/tools/check_doc_anchors.py`, run by `build_docs.sh`, proves the other
  half end to end: for every cited anchor it looks in Doxygen's XML output for
  the section id Doxygen actually generated.

The generated **Evidence index** page (`@ref evidence_index`) is the reverse
map: for each cited section, every file and line that depends on it. Check it
before rewording or deleting a section. When you must rename a cited heading,
update every pointer in the same change; the checker makes a half-done rename a
red build.

## Writing API documentation

API documentation lives in the public headers (`include/batchlas/`,
`include/batchlas.hh`) and describes the **contract**: what the function
computes, each parameter, the return value, preconditions, what it throws, and
complexity or workspace requirements where they matter to a caller. It does not
describe why the implementation is shaped the way it is; that is a design or
evidence page, linked with `@see` or an evidence pointer.

Style:

```cpp
/// @brief One-line summary ending with a full stop.
///
/// Optional longer description of the contract.
/// @param queue  queue the kernels are enqueued on
/// @param a      batch of n x n matrices; overwritten with the factor
/// @return       event of the last enqueued kernel
/// @pre  a.rows() == a.cols()
/// @throws batchlas::Error if the workspace is too small
/// @ingroup factorizations
```

`/** ... */` and `/*! ... */` blocks are equivalent. `JAVADOC_AUTOBRIEF` is on,
so the first sentence is the brief even without `@brief`. Use `///<` for a
trailing member comment. The Doxyfile defines three aliases: `@invariant`,
`@trap` and `@evidence{label}`.

Every public entity belongs to a group. The group ids are defined in
`docs/pages/api_groups.dox`: `core`, `matrix`, `enums`, `options`, `blas2`,
`blas3`, `factorizations`, `qr`, `eigen`, `tridiag`, `svd`, `sparse`, `extra`,
`linalg`, `dispatch`, `workspace`, `errors`, `config`, `device` and
`internal_helpers`, all nested under `api`. Add `@ingroup <id>` to the
declaration, or wrap a run of declarations in `@addtogroup <id>` ... `@{` ...
`@}`. Add a new group to `api_groups.dox` rather than inventing one in a header.

**The comment-density exemption.** In the public headers, comment-only lines
that belong to a Doxygen doc comment (`///`, `//!`, or a block opened by `/**`
or `/*!`) are counted in neither the numerator nor the denominator of the 18%
ceiling; `check_comment_density.py --all` reports them as `doc`. A `/***`
banner, an empty `/**/` and a `////` rule are ordinary comments, and a line
that mixes a doc comment and a plain one counts. The same doc comments in
`src/`, `tests/` or `benchmarks/` count normally, because Doxygen does not
publish those. The exemption is for the contract only: rationale, measurements
and design history written with `///` are still rationale, and review rejects
them. Move them to `docs/` and leave a pointer.

## Adding results

1. **Raw data** goes in `benchmarks/results/`, which is tracked by Git LFS. Run
   `git lfs install` once per machine before committing, or the file is
   committed raw and `.github/ci/check_lfs_pointers.py` fails CI. Name the file
   `<campaign>_<op>_<dtype>[_<machine>].csv` where you can; the generator
   reads the op, type and machine out of the name. It knows one machine token,
   `rtx4090`; a file from another machine needs its token added to `MACHINES`
   in `docs/tools/gen_db_pages.py`, or its row claims the primary box.
2. **The evidence section** that uses the data records, next to the numbers:
   the machine (GPU and SM, and the host if it matters), the date of the
   measurement, the command or harness with its flags (batch, warm-up,
   iterations, pinned route), and the path of the raw file. A ratio without its
   machine and date cannot be re-checked when the vendor library moves. Follow
   the measurement rules on @ref perf_evidence.
3. **The Results database** page (`@ref results_database`) is generated on
   every build from `benchmarks/results/`: one row per file, with the campaign,
   op, type and machine parsed from the name, the date and commit that added
   it, whether this checkout holds the data or only the LFS pointer, and which
   `docs/` pages mention the file by name. A grid that no page mentions shows
   `—` in the last column, which means it is not yet distilled anywhere.
4. Older raw evidence is at the tag `perf-evidence/vendor-independence`
   (`git show perf-evidence/vendor-independence:experiments/<path>`); pages cite
   those paths as archive paths.

## Page template

````md
# <op or topic>: <what the page is about> {#perf_<op>}

> **Covers:** <one sentence: which ops, routes or decisions>.
> **Status:** current | historical (superseded by @ref <label>).
> **Machine:** RTX 4090 (sm_89), CUDA 13.2, /opt/dpcpp-cuda, unless a section says otherwise.
> **Measured:** <dates as recorded in the source material>.

<Two or three sentences: the conclusion a reader needs first.>

## <op>: the shipped predicate

<Quote the predicate with file:line. Read the predicate, not the prose.>

## <op>: <distinctive name of the window or decision>

<Numbers, with the bracketing non-winner on each edge. Raw data:
benchmarks/results/<file>.csv. Command: <exact command>.>

## <op>: rejected alternatives

<What was tried, the measured result, and why it lost.>

## <op>: open debts

<What is still owed and what would settle it.>
````

For a design, algorithm, guide or developer page use the matching label prefix
from [Where things go](#where-things-go); drop the Machine and Measured lines
when the page holds no measurements.

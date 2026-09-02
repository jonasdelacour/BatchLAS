# Routing and performance evidence

Every native kernel in BatchLAS competes with a vendor library, and the choice between them
is a **measured window**, not a preference. These pages are the record of those measurements:
what each op's `preferred()` predicate actually is, the grid that justifies each of its
boundaries, what was built and rejected, and what is still owed.

Read the page for an op before you widen its window, add a tier, or "fix" a route that looks
conservative. Most of the obvious moves in here have already been made and measured worse.

| page | ops | does anything route natively by default? |
|---|---|---|
| [dispatch.md](dispatch.md) | the `Route` vocabulary, the vendor gate, the coverage instrument | n/a — this is the mechanism |
| [gemm.md](gemm.md) | `gemm` | **yes** — `double` broadly, `float` NN squares at `max_dim <= 32`; complex never |
| [level3.md](level3.md) | `symm` `hemm` `syrk` `herk` `syr2k` `her2k` `trmm` | hand-rolled `if`-chains in the facade, not route tables |
| [trsm.md](trsm.md) | `trsm` | **yes**, broadly — but see its open debts before trusting a ratio |
| [potrf.md](potrf.md) | `potrf` | no — two native tiers ship, neither is preferred |
| [qr.md](qr.md) | `geqrf` `orgqr` `ormqr` | `ormqr` yes; `geqrf`/`orgqr` route-neutral |
| [lu.md](lu.md) | `getrf` `getrs` `getri` | **yes** — four windows, all `float`/`cfloat`-leaning |
| [gemv.md](gemv.md) | `gemv` | **yes** — one `complex<double>` transposed window |
| [spmm.md](spmm.md) | `spmm` | **yes** — the `NoTrans` gather; the transposed scatter stays vendor-first |

## Two rules these pages are written to

**The shipped code is the authority on *what* ships; the notes are the authority on *why*.**
Several of these windows were narrowed or widened after the note describing them was written,
and a few in-tree comments are still stale against their own predicate — `lu.md` lists ten
source locations that claim `preferred()` is all-false when four windows now ship. Every page
quotes the predicate with a `file:line`. Read the predicate, not the prose above it.

**`supports()` and `preferred()` are not the same kind of false.** `supports()` is
correctness: false means the route would return a *wrong answer*. `preferred()` is a measured
window: false means merely *slower*, and the route stays eligible — vendor-free, an
un-preferred native route is still the one that runs. Putting a performance threshold in
`supports()` silently disables the vendor-free fallback and makes "forced native" tests run
the vendor and pass green. That mistake is documented in `potrf.md`.

## Measurement rules

Ratios in these pages were taken under the following, and a page says so where it deviates:

- **At saturation.** An unsaturated ratio measures overhead, not the algorithm. Where an arm
  is below saturation the page says so beside the number.
- **One harness on the box at a time.** This machine has two RTX 4090s on one NUMA node
  sharing a UVM driver; a sweep on device 1 alongside a sweep on device 0 read one cell 5.5×
  slow while `nvidia-smi --query-compute-apps` reported zero foreign processes on both cards
  and `rel_sd` stayed under 0.02. It is cell-specific and intermittent. See `lu.md`.
- **JIT warmed**, medians of interleaved A/B, cells over a relative-sd gate discarded rather
  than reported.
- **Vendor-free means the build** (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`), never an environment
  variable inside a build that still links the vendor: an unsupported forced route falls
  through to `automatic()`, which returns `{Vendor, Auto}`, so a forced-route A/B inside one
  build can silently be vendor-against-vendor.
- **Every boundary wants a bracketing non-winner.** A window edge with no measured loss just
  outside it is a guess. Where one is missing the page marks it.

## The raw data

The 4,771 files and ~13 MB of CSV, logs, profiler captures, drivers and standalone harnesses
behind these pages are preserved verbatim at the tag **`perf-evidence/vendor-independence`**,
which is the tree as it stood at the end of WP8. They were removed from the working tree
because they were 179k lines against 50k lines of actual change, and no build or test reads
them.

Each page's final section maps its claims to the paths that hold their raw data. Retrieve one
with:

```
git show perf-evidence/vendor-independence:experiments/wp6_lu/bench/README.md
git show perf-evidence/vendor-independence:experiments/sparse_spmm/verdict.txt
```

or check the whole tree out somewhere scratch:

```
git worktree add /tmp/perf-evidence perf-evidence/vendor-independence
```

## Related

- [../design/vendor-independence.md](../design/vendor-independence.md) — how dispatch works
  and how to add an op to it.
- [../design/vendor-free-status.md](../design/vendor-free-status.md) — what the vendor-free
  build currently does and does not do.
- [../design/known-defects.md](../design/known-defects.md) — located, unfixed, with line
  numbers.

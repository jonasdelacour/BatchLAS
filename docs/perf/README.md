# Routing and performance evidence {#perf_evidence}

Every native kernel competes with a vendor library, and the choice between them is a measured
window. These pages record the measurements behind each window: the grid that sets each boundary,
the alternatives that were rejected, and the open debts. Read an op's page before widening its
window or adding a tier.

Since flat selection ([../design/flat-kernel-selection.md](../design/flat-kernel-selection.md)),
each op ranks its kernels from `tuned/<op>.<dtype>.<device>.txt`. Where a page quotes a
`preferred()` or `supports()` predicate with its source line number, that predicate is deleted. Its window
lives on as table rows.

| Page | Ops | Does anything route natively by default? |
|---|---|---|
| [dispatch.md](dispatch.md) | the `BATCHLAS_<OP>_ROUTE` words, the vendor gate, the coverage instrument, level-3 boundaries | n/a (the mechanism) |
| [gemm.md](gemm.md) | `gemm` | **yes**: `double` broadly, `float` NN squares at `max_dim <= 32`; complex never |
| [level3.md](level3.md) | `symm` `hemm` `syrk` `herk` `syr2k` `her2k` `trmm` | symm, syrk, syr2k, trmm: flat selection tables; hemm, herk, her2k: hand-written in `cublas.cc` |
| [trsm.md](trsm.md) | `trsm` | **yes**, broadly; read its open debts before trusting a ratio |
| [potrf.md](potrf.md) | `potrf` | **yes**, per cell; measured tables; five native kernels |
| [qr.md](qr.md) | `geqrf` `orgqr` `ormqr` | **yes**: `ormqr` native-first; `geqrf` above a per-type order floor plus a tall-panel clause; `orgqr` to n = 512 |
| [lu.md](lu.md) | `getrf` `getrs` `getri` | **yes**: four windows, all `float`/`cfloat`-leaning |
| [gemv.md](gemv.md) | `gemv` | **yes**: one `complex<double>` transposed window |
| [spmm.md](spmm.md) | `spmm` | **yes**: the `NoTrans` gather; the transposed scatter stays vendor-first |
| [steqr.md](steqr.md) | `steqr_cta` (also the fused small-n `syev` and the `stedc` leaves) | n/a; no vendor arm |
| [syev.md](syev.md) | `syev` (Jacobi CTA, fused CTA, blocked, two-stage; vendor) | **yes**: native small-n and blocked windows per type |
| [syevx.md](syevx.md) | `syevx` (Direct, DirectSubset, Filtered, LOBPCG, stebz/stein) | n/a; no vendor arm for the selected-range solve |
| [stedc.md](stedc.md) | `stedc` (divide and conquer, merge kernels) | n/a; no vendor arm |
| [sytrd.md](sytrd.md) | `sytrd_blocked` + `latrd`, `sytrd_sy2sb`, `sytrd_sb2st_hh` and its Q2 back-transform | n/a; internal reductions under `syev` |
| [ortho.md](ortho.md) | `ortho` (the Gram product and the algorithm rules) | n/a; the Gram product goes through `syrk` (real, small `k`) or `gemm`; host devices force Householder |
| [iluk.md](iluk.md) | `iluk` (ILU(k) numeric phase and apply) | n/a; host or device chosen on batch size |
| [tuning.md](tuning.md) | the `tuning_params.hh` constants and `BATCHLAS_TUNE_*` overrides | n/a; kernel parameters, not a routing choice |
| [gesvd.md](gesvd.md) | `gesvd` | **yes**: `jacobi` for real general max(m,n) <= 32 and complex general input up to its ceiling; `blocked` above for real general; `cta` for square Hermitian to n = 32 |

## perf index: two rules

**The shipped code decides what ships; the notes explain why.** Several windows were narrowed or
widened after their note was written. Read the op's table and its `can_run` in
`src/ops/<op>/<op>.cc`. Use the select trace (`BATCHLAS_SELECT_TRACE`) or the coverage `reached`
row to see which kernel a shape takes.

**`can_run` and table rows are different kinds of false.** `can_run` is correctness: false means
the kernel would refuse or return a wrong answer. A table row is speed: a candidate ranked last is
slower but stays eligible. A speed threshold in `can_run` makes the shape unservable vendor-free and
turns a pin into a throw.

## perf index: measurement rules

- **At saturation.** Below saturation a ratio measures overhead. Where an arm is below saturation,
  its page says so beside the number.
- **One harness on the box at a time.** The two RTX 4090s share one NUMA node and one UVM driver. A
  sweep on device 1 alongside one on device 0 read a cell 5.5x slow, while
  `nvidia-smi --query-compute-apps` showed no foreign process and `rel_sd` stayed under 0.02. The
  effect is cell-specific and intermittent; see `lu.md`.
- **JIT warmed.** Medians of interleaved A/B. Cells over a relative-sd gate are discarded.
- **Vendor-free means the build** (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`), never an environment
  variable inside a build that still links the vendor. There, an unsupported forced route falls
  through to `automatic()`, which returns `{Vendor, Auto}`, so an A/B can be vendor against vendor.
- **Every boundary needs a bracketing non-winner.** A window edge with no measured loss just outside
  it is a guess. Where one is missing, the page says so.

## perf index: the raw data

The raw data behind the older pages (about 4,800 files and 13 MB of CSV, logs, profiler captures and
harnesses) is preserved at the tag **`perf-evidence/vendor-independence`**. No build or test reads it,
so it is out of the working tree. Every `experiments/...` path on these pages is an archive path,
and resolves only against the tag:

```
git show perf-evidence/vendor-independence:experiments/wp6_lu/bench/README.md
git show perf-evidence/vendor-independence:experiments/sparse_spmm/verdict.txt
git worktree add /tmp/perf-evidence perf-evidence/vendor-independence
```

Each page's final section maps its claims to the paths that hold their data.

### New raw data lives in `benchmarks/results/`, in Git LFS

New grids go under `benchmarks/results/`. `.gitattributes` routes that directory through Git LFS, so
the repository stores a three-line pointer per file. A PR that adds a grid costs about three lines
per file in its diff. File names, paths, citations and scripts are unchanged.

```
git lfs install        # once per machine
git lfs pull           # fetch the content for the current checkout
```

Committing without `git-lfs` installed stores the raw file silently. `.github/ci/check_lfs_pointers.py`
catches this: it fails CI if an LFS-tracked blob is not a pointer, and `.github/ci/run_local_checks.sh`
runs it against the index before the commit. The fix is `git lfs install`, then
`git add --renormalize benchmarks/results`. No data is lost.

## perf index: related pages

- [../design/vendor-independence.md](../design/vendor-independence.md): how dispatch works and how to
  add an op to it.
- [../design/vendor-free-status.md](../design/vendor-free-status.md): what the vendor-free build does
  and does not do.
- [../design/known-defects.md](../design/known-defects.md): located, unfixed defects, with line numbers.

## perf index: all evidence pages

@subpage md_docs_2perf_2dispatch "Dispatch"

@subpage perf_gemm "GEMM"

@subpage perf_gemv "GEMV"

@subpage md_docs_2perf_2level3 "Level 3 (symm, hemm, syrk, herk, syr2k, her2k, trmm)"

@subpage perf_trsm "TRSM"

@subpage md_docs_2perf_2potrf "Cholesky (potrf, posv)"

@subpage perf_lu "LU (getrf, getrs, getri, gesv)"

@subpage md_docs_2perf_2qr "QR (geqrf, orgqr, ormqr)"

@subpage md_docs_2perf_2spmm "SpMM"

@subpage perf_syev "syev"

@subpage perf_syevx "syevx"

@subpage perf_steqr "steqr"

@subpage perf_stedc "stedc"

@subpage perf_sytrd "sytrd (tridiagonal and band reduction)"

@subpage perf_ortho "ortho"

@subpage perf_iluk "ILU(k)"

@subpage perf_gesvd "gesvd"

@subpage md_docs_2perf_2small-n-baseline "Small-n factorization baseline"

@subpage perf_tuning "Tuning constants"

@subpage md_docs_2perf_2blackwell "Blackwell (sm_120) retune"

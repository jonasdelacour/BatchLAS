# Routing and performance evidence {#perf_evidence}

Every native kernel competes with a vendor library, and the choice between them is measured. These
pages hold the measurements behind each routing window, tuning constant and design decision, the
rejected alternatives, and the open debts. Read an op's page before widening its window or adding a
tier. Each op ranks its kernels from `tuned/<op>.<device>.txt`
([flat selection](../design/flat-kernel-selection.md)); a `preferred()` or `supports()` predicate quoted
on an older page no longer exists, its window lives on as table rows.

| Page | Ops |
|---|---|
| [dispatch.md](dispatch.md) | the `BATCHLAS_<OP>_ROUTE` words, buffer-size queries, coverage instrument, dispatch bugs |
| [gemm.md](gemm.md), [gemv.md](gemv.md), [level3.md](level3.md), [trsm.md](trsm.md) | `gemm`, `gemv`, `symm` `hemm` `syrk` `herk` `syr2k` `her2k` `trmm`, `trsm` |
| [potrf.md](potrf.md), [potri.md](potri.md), [potrs.md](potrs.md) | `potrf` `posv`, `potri`, `potrs` |
| [lu.md](lu.md), [sytrf.md](sytrf.md), [qr.md](qr.md) | `getrf` `getrs` `getri`, `sytrf` `sytrs` `sysv`, `geqrf` `orgqr` `ormqr` |
| [small-n-baseline.md](small-n-baseline.md) | small-n factorizations against the vendor |
| [syev.md](syev.md), [sytrd.md](sytrd.md), [steqr.md](steqr.md), [stedc.md](stedc.md) | `syev` and its reductions and tridiagonal solvers |
| [syevx.md](syevx.md), [sygvd.md](sygvd.md), [syfn.md](syfn.md), [gesvd.md](gesvd.md) | selected-range, generalized, matrix-function and SVD solvers |
| [spmm.md](spmm.md), [ortho.md](ortho.md), [iluk.md](iluk.md) | `spmm`, `ortho`, `iluk` |
| [tuning.md](tuning.md), [blackwell.md](blackwell.md) | the `tuning_params.hh` constants and `BATCHLAS_TUNE_*`; the sm_120 retune |
| [batch-sensitivity.md](batch-sensitivity.md) | why per-item time depends on batch, and the cold protocol |
| [sycl-implementations.md](sycl-implementations.md) | DPC++ against AdaptiveCpp, performance and accuracy |

## perf index: two rules

**The shipped code decides what ships; the notes explain why.** Read the op's table and its `can_run`
in `src/ops/<op>/`; `BATCHLAS_SELECT_TRACE=1` or the coverage `reached` row shows which kernel a shape
takes.

**`can_run` and table rows are different kinds of false.** `can_run` is correctness: false means the
kernel would refuse or return a wrong answer. A table row is speed: a candidate ranked last stays
eligible. A speed threshold in `can_run` makes the shape unservable vendor-free and turns a pin into
a throw.

## perf index: measurement rules

- **At saturation** (batch >= 128). Below it a ratio measures overhead; a page says so beside such a number.
- **One harness on the box at a time.** A sweep on the other 4090 once read a cell 5.5x slow with no
  foreign process visible and `rel_sd` under 0.02.
- **JIT warmed**, medians of interleaved A/B, cells over a relative-sd gate discarded; the tuner
  scrubs L2 before every sample ([cold protocol](batch-sensitivity.md#batch-sensitivity-the-cold-protocol)).
- **Vendor-free means the build** (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`), never an environment variable:
  in a vendor build the class word `native` with nothing runnable falls back to Auto, which may pick the
  vendor.
- **Every boundary needs a bracketing non-winner.** A window edge with no measured loss just outside it
  is a guess; where one is missing, the page says so.

The full protocol is the agent guide's §10 (@ref dev_agent_guide).

## perf index: the raw data

Raw data behind the older pages (CSV, logs, profiler captures, harnesses) is preserved at the tag
**`perf-evidence/vendor-independence`**, outside the working tree. Every `experiments/...` path on
these pages resolves only against it:

```
git show perf-evidence/vendor-independence:experiments/wp6_lu/bench/README.md
git worktree add /tmp/perf-evidence perf-evidence/vendor-independence
```

### New raw data lives in `benchmarks/results/`, in Git LFS

`.gitattributes` routes `benchmarks/results/` through Git LFS (`git lfs install` once, `git lfs pull`
to fetch). Committing without `git-lfs` stores the raw file; `.github/ci/check_lfs_pointers.py` fails
CI on it. Fix with `git lfs install`, then `git add --renormalize benchmarks/results`.

## perf index: related pages

- [../design/vendor-independence.md](../design/vendor-independence.md): how dispatch works and how to
  add an op to it.
- [../design/vendor-free-status.md](../design/vendor-free-status.md): what the vendor-free build does.
- [../design/known-defects.md](../design/known-defects.md): located, unfixed defects.

## perf index: all evidence pages

@subpage perf_dispatch "Dispatch"

@subpage perf_gemm "GEMM"

@subpage perf_gemv "GEMV"

@subpage perf_level3 "Level 3 (symm, hemm, syrk, herk, syr2k, her2k, trmm)"

@subpage perf_trsm "TRSM"

@subpage perf_potrf "Cholesky (potrf, posv)"

@subpage perf_potri "potrs and potri"

@subpage perf_potrs "potrs"

@subpage perf_lu "LU (getrf, getrs, getri, gesv)"

@subpage perf_sytrf "LDL^T (sytrf, sytrs, sysv)"

@subpage perf_qr "QR (geqrf, orgqr, ormqr)"

@subpage perf_spmm "SpMM"

@subpage perf_syev "syev"

@subpage perf_syevx "syevx"

@subpage perf_sygvd "sygvd (generalized eigenproblem)"

@subpage perf_syfn "syfn (matrix functions)"

@subpage perf_steqr "steqr"

@subpage perf_stedc "stedc"

@subpage perf_sytrd "sytrd (tridiagonal and band reduction)"

@subpage perf_ortho "ortho"

@subpage perf_iluk "ILU(k)"

@subpage perf_gesvd "gesvd"

@subpage perf_small_n_baseline "Small-n factorization baseline"

@subpage perf_tuning "Tuning constants"

@subpage perf_blackwell "Blackwell (sm_120) retune"

@subpage perf_batch_sensitivity "Batch-size sensitivity"

@subpage perf_sycl_implementations "DPC++ vs AdaptiveCpp"

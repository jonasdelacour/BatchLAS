# BatchLAS Python examples

Twelve self-checking notebooks for the `batchlas` Python package. Each one
explains a slice of the API in prose, then computes something and verifies it
against NumPy/SciPy — so a clean run doubles as a smoke test:

```
   [ok  ] |Q^T Q - I|: 4.441e-16  (tol 1.0e-08)
```

Lines marked `[FAIL]` mean a check did not hold on your machine.

The notebooks are committed with output from a reference run, so they render
fully on GitHub without executing anything.

## Running them

The notebooks import `batchlas`, so it must be importable by the kernel. From a
build tree with `BATCHLAS_BUILD_PYTHON=ON`:

```bash
cmake -B build -S . -DBATCHLAS_BUILD_PYTHON=ON
cmake --build build -j

cd python/examples
PYTHONPATH=../../build/python jupyter lab
```

To execute and check all twelve without opening Jupyter:

```bash
PYTHONPATH=../../build/python python3 run_all.py
PYTHONPATH=../../build/python python3 run_all.py 05 06   # just these
```

`run_all.py` exits non-zero if any notebook raises or reports a failed check.
After editing a notebook, `--save` re-executes it and writes the refreshed
output back in, so the committed rendering matches the code:

```bash
PYTHONPATH=../../build/python python3 run_all.py 07 --save
```

Requirements: NumPy for everything, SciPy for notebook 09, and
`nbformat` + `nbclient` + `ipykernel` for `run_all.py`. A GPU is optional — the
notebooks fall back to whatever device the library picks — but a few routines
are GPU-only, as noted below.

## The notebooks

| # | Notebook | What it covers |
|---|------|----------------|
| 01 | `01_getting_started.ipynb` | Backends, devices, features, first `gemm`, the batching convention, dtypes, `out=` |
| 02 | `02_dense_blas.ipynb` | `gemm`, `gemv`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm`, heterogeneous batches, mixed precision |
| 03 | `03_linear_solvers.ipynb` | `potrf`, `getrf`/`getrs`, `getri`, `inv`, triangular solves, complex input |
| 04 | `04_qr_and_orthogonalization.ipynb` | `geqrf`, `orgqr`, `ormqr`, `ortho` algorithms, `ortho_metric` |
| 05 | `05_svd.ipynb` | `gesvd`, `gesvd_blocked`, `gesvd_cta`, `gebrd_*`, `bdsqr`, `ormbr` |
| 06 | `06_symmetric_eigensolvers.ipynb` | The whole `syev` family incl. `syev_jacobi_cta`, `syev_variant_support`, options objects |
| 07 | `07_tridiagonal_reduction.ipynb` | `sytrd_cta`, `sytrd_blocked`, `sytrd_sy2sb`, `sytrd_sb2st`, `hetrd_hb2st`, `sytrd_band_reduction` |
| 08 | `08_tridiagonal_eigensolvers.ipynb` | `steqr`, `steqr_cta`, `stedc`, `stedc_flat`, `tridiagonal_solver` |
| 09 | `09_sparse_and_iterative.ipynb` | `spmm`, `syevx` (+ convergence history), `lanczos`, `ritz_values`, ILU(k) |
| 10 | `10_jacobi_relative_accuracy.ipynb` | Why `syev_jacobi_cta` exists: relative accuracy on graded matrices |
| 11 | `11_generators_and_utilities.ipynb` | Constructors, conditioned random generators, `norm`, `cond`, `transpose`, `lascl` |
| 12 | `12_choosing_a_variant.ipynb` | Batching speed-up, throughput scaling, picking a `syev` variant, CPU vs GPU |

Between them they exercise 77 of the 78 names exported by `batchlas` (the 78th is
`ILUKPreconditioner`, the handle type returned by `iluk_factorize`).

`_common.py` holds shared helpers (device selection, reporting, reference
constructions) that the notebooks import. It is not part of the library API.

Timings and device names in notebook 12 come from a run on an RTX 4090 and will
differ on your hardware.

## Conventions worth knowing

- **Batching.** A 2-D array is one matrix; a 3-D array of shape
  `(batch, rows, cols)` is a batch. The same call handles both. Pass a *list* of
  2-D arrays for a heterogeneous batch, where shapes may differ.
- **dtypes.** `float32`, `float64`, `complex64`, `complex128`. Output dtype
  follows input dtype.
- **`device=` and `backend=`.** Both default to letting the library choose.
  `device` is `"cpu"`, `"gpu"`, `"accelerator"`, or `None`.
- **`out=`.** Where a routine accepts it, `out=` is both the destination buffer
  and the `C` operand of a BLAS update. `beta != 0` therefore requires `out=`;
  the call raises a clear `ValueError` otherwise.
- **`uplo`.** Symmetric routines read only the nominated triangle. Passing the
  full symmetric matrix is always safe.
- **Options objects.** Tuning parameters are dataclasses (`SteqrOptions`,
  `StedcOptions`, `JacobiOptions`, `SyevxOptions`, `LanczosOptions`,
  `ILUKOptions`, `SytrdBandReductionOptions`). A plain dict works too.
- **Tridiagonal input.** `(d, e)` with `len(d) == n` and `len(e) == n - 1`.
- **Band storage.** `(kd + 1, n)`, lower LAPACK convention: `AB[i, j]` holds
  `A[j + i, j]`. `_common.band_to_dense` expands it.

## Known issues visible in these notebooks

These are library-level problems, not mistakes in the examples. They are called
out in the notebooks where they come up.

- **`ortho(algorithm="householder")` on CUDA.** Returns a non-orthonormal result
  if any earlier call in the same process consumed a `geqrf` workspace.
  `ortho_buffer_size` sizes its `geqrf`/`orgqr` sub-workspaces from a placeholder
  view rather than the real `A` (`src/extensions/ortho.cc`), so the bump
  allocator can hand out overlapping blocks. The other algorithms are unaffected;
  notebook 04 uses those.
- **`stedc` / `stedc_flat` with `JobType::NoEigenVectors`.** Returns wrong
  eigenvalues, and slices an eigenvector output it was told not to produce. The
  Python bindings work around this by always requesting vectors internally and
  discarding them, which is also what `syev_blocked` does inside the library. The
  same defect is why `syev_two_stage(compute_vectors=False)` needed the same
  workaround.
- **`stedc_flat` eigenvectors.** Eigenvalues are correct; the eigenvectors do not
  satisfy `A V = V diag(w)`. Notebook 08 reports this residual without a
  tolerance so it stays visible.
- **`tridiagonal_solver` accuracy.** Its QR iteration does not converge reliably;
  accuracy varies with `n` and with the data. Prefer `steqr` or `stedc`.
- **`uplo="upper"` with a half-filled matrix on CUDA.** `syev` and `syev_cta`
  give the right answer for a full symmetric matrix and for `uplo="lower"` with
  only the lower triangle filled, but not for `uplo="upper"` with only the upper
  triangle filled. `syev_jacobi_cta` and the CPU path handle it correctly.
- **Unpreconditioned `syevx` on hard problems.** Can stagnate rather than
  converge; notebook 09 shows both the stalling case and the preconditioned fix.

## Device requirements

The `*_cta` routines map one work-group onto one matrix and need a sub-group
width of 32, so they are GPU-only; on CPU they raise
`RuntimeError: ... device does not support subgroup size 32 ...`. They are also
limited to `n <= 32`. Call `syev_variant_support(a, device=...)` to ask the
device directly instead of guessing.

# BatchLAS Python examples {#python_examples}

Twelve self-checking notebooks for the `batchlas` Python package. Each explains part of the API,
computes something, and checks the result against NumPy/SciPy, so a clean run doubles as a smoke
test:

```
   [ok  ] |Q^T Q - I|: 4.441e-16  (tol 1.0e-08)
```

A `[FAIL]` line means a check did not hold on your machine. The notebooks are committed with output
from a reference run, so they render on GitHub without executing.

## Running them

The notebooks import `batchlas`, so the kernel must be able to find it. From a build tree configured
with `BATCHLAS_BUILD_PYTHON=ON`:

```bash
cmake -B build -S . -DBATCHLAS_BUILD_PYTHON=ON
cmake --build build -j

cd python/examples
PYTHONPATH=../../build/python jupyter lab
```

To execute and check all twelve without Jupyter:

```bash
PYTHONPATH=../../build/python python3 run_all.py
PYTHONPATH=../../build/python python3 run_all.py 05 06   # just these
```

`run_all.py` exits non-zero if a notebook raises or reports a failed check. After editing a
notebook, `--save` re-executes it and writes the refreshed output back:

```bash
PYTHONPATH=../../build/python python3 run_all.py 07 --save
```

Requirements: NumPy for all notebooks, SciPy for notebook 09, and `nbformat`, `nbclient` and
`ipykernel` for `run_all.py`. A GPU is optional. The notebooks fall back to the device the library
picks, but some routines are GPU-only (see Device requirements below).

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
| 08 | `08_tridiagonal_eigensolvers.ipynb` | `steqr`, `steqr_cta`, `stedc`, `tridiagonal_solver` |
| 09 | `09_sparse_and_iterative.ipynb` | `spmm`, `syevx` (with convergence history), `lanczos`, `ritz_values`, ILU(k) |
| 10 | `10_jacobi_relative_accuracy.ipynb` | Why `syev_jacobi_cta` exists: relative accuracy on graded matrices |
| 11 | `11_generators_and_utilities.ipynb` | Constructors, conditioned random generators, `norm`, `cond`, `transpose`, `lascl` |
| 12 | `12_choosing_a_variant.ipynb` | Batching speed-up, throughput scaling, picking a `syev` variant, CPU vs GPU |

Together they exercise 77 of the 78 names exported by `batchlas`. The 78th is `ILUKPreconditioner`,
the handle type returned by `iluk_factorize`.

`_common.py` holds shared helpers (device selection, reporting, reference constructions). It is not
part of the library API.

Timings and device names in notebook 12 come from an RTX 4090 and will differ on your hardware.

## Conventions

- **Batching.** A 2-D array is one matrix. A 3-D array of shape `(batch, rows, cols)` is a batch.
  The same call handles both. Pass a *list* of 2-D arrays for a heterogeneous batch, where shapes
  may differ.
- **dtypes.** `float32`, `float64`, `complex64`, `complex128`. The output dtype follows the input.
- **`device=` and `backend=`.** Both default to letting the library choose. `device` is `"cpu"`,
  `"gpu"`, `"accelerator"` or `None`.
- **`out=`.** Where a routine accepts it, `out=` is both the destination buffer and the `C` operand
  of a BLAS update. `beta != 0` therefore requires `out=`, and the call raises `ValueError` otherwise.
- **`uplo`.** Symmetric routines read only the nominated triangle. Passing the full symmetric matrix
  is always safe.
- **Options objects.** Tuning parameters are dataclasses: `SteqrOptions`, `StedcOptions`,
  `JacobiOptions`, `SyevxOptions`, `LanczosOptions`, `ILUKOptions`, `SytrdBandReductionOptions`.
  A plain dict also works.
- **Tridiagonal input.** `(d, e)` with `len(d) == n` and `len(e) == n - 1`.
- **Band storage.** `(kd + 1, n)`, lower LAPACK convention: `AB[i, j]` holds `A[j + i, j]`.
  `_common.band_to_dense` expands it.

## Known issues

These are library defects. The notebooks call them out where they come up.

- **`ortho(algorithm="householder")` on CUDA.** After any earlier `geqrf` workspace use in the same
  process, returns a non-orthonormal result. `ortho_buffer_size` sizes its sub-workspaces from a
  placeholder view (`src/extensions/ortho.cc`), so blocks can overlap. Other algorithms are
  unaffected; notebook 04 uses those.
- **`stedc` with `JobType::NoEigenVectors`.** Returns wrong eigenvalues. The bindings always request
  vectors internally and discard them.
- **`tridiagonal_solver` accuracy.** The QR iteration does not converge reliably. Prefer `steqr` or
  `stedc`.
- **`uplo="upper"` with a half-filled matrix on CUDA.** `syev` and `syev_cta` are wrong for this
  case. `syev_jacobi_cta` and the CPU path are correct.
- **Unpreconditioned `syevx` on hard problems.** Can stagnate. Notebook 09 shows the fix.

## Device requirements

The `*_cta` routines map one work-group onto one matrix. They need a sub-group width of 32, so they
are GPU-only. On a CPU they raise `RuntimeError: ... device does not support subgroup size 32 ...`.
They also accept only `n <= 32`. To ask the device directly instead of guessing, call
`syev_variant_support(a, device=...)`.

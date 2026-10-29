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
cmake --build build --target _batchlas -j8

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

`_common.py` holds shared helpers, not library API. Notebook 12's timings are from an RTX 4090.

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

## Not in the notebooks

Checked against NumPy/SciPy in `python/tests/` instead.

- **Solvers and Hermitian level 3.** `gesv` (LU solve) and `posv` (Cholesky solve), both with
  `return_factors=`; `hemm`, `herk`, `her2k`.
- **Per-item status.** `return_info=True` on `potrf`, `gesv`, `posv`, `getri`, `gesvd`, `syev` and
  `syevx` appends an int32 `info` array, one entry per item: `0` is success, `i > 0` as LAPACK.
- **Thin SVD.** `gesvd(a, compute_vectors="thin")`: `u` is `m x k`, `vh` is `k x n`, `k = min(m, n)`.
- **Exceptions.** `BatchLASException` is the base; the classes and their builtin bases are in
  [Python exception classes](../../docs/design/error-model.md#error-model-python-exception-classes).
- **Routes.** `set_route(op, spelling)` pins `BATCHLAS_<OP>_ROUTE` for this process (`None`
  unpins), `route(op, spelling)` is the context-manager form, `route_pin(op)` reads the pin, and
  `reload_settings()` re-reads every `BATCHLAS_*` variable changed after the first call.

## `batchlas.linalg`

A NumPy/SciPy-flavoured front end of the C++ `batchlas::linalg` layer (`import batchlas.linalg`
or `bl.linalg`). Unlike the top-level functions:

- Any leading axes are the batch, and every output keeps them. Per-item scalars (`info`, `count`,
  `norm`, `cond`) have the batch shape, so a 2-D input gives a NumPy scalar.
- Integer input is promoted to float64; two operands are promoted to their common dtype.
- Factorisations and solves always return a named tuple ending in `info` (the C++ layer always
  fills it). A failed item is reported there, never raised: check `np.all(r.info == 0)`.
- `lower=True` (default) names the triangle that is read.

| Function | Returns |
|---|---|
| `matmul(a, b, alpha=, trans_a=, trans_b=, compute_precision=)` | array |
| `matvec(a, x, trans=)` | array |
| `gram(a, trans="c")` | \f$A^HA\f$; `trans="n"` gives \f$AA^H\f$ |
| `solve_triangular(a, b, lower=, trans=, unit_diagonal=, side=, alpha=)` | array; `trans` also SciPy's 0/1/2 |
| `cholesky(a, lower=)` | `CholeskyResult(factor, info)` |
| `solve(a, b, trans=)`, `solve_spd(a, b, lower=)` | `SolveResult(x, info)`; a vector `b` (`b.ndim == a.ndim - 1`) is a column |
| `inv(a)` | `InvResult(inverse, info)` |
| `lu(a)` | `LUResult(lu, pivots, info)`; 1-based LAPACK pivots, as `getrf` |
| `eigh(a, lower=)` | `EighResult(eigenvalues, eigenvectors, info)` |
| `eigvalsh(a, lower=)` | array |
| `eigh`/`eigvalsh` with `subset_by_index=(il, iu)` or `subset_by_value=(vl, vu)` | `EighSubsetResult(eigenvalues, eigenvectors, count, info)` / `EigvalshSubsetResult(eigenvalues, count, info)` |
| `svd(a, full_matrices=, compute_uv=, hermitian_uplo=, jobu=, jobvh=)` | `SVDResult(U, S, Vh, info)` |
| `triu(a, k)`, `tril(a, k)`, `transpose(a)` | array |
| `norm(a, ord="fro")`, `cond(a, ord="fro")` | per-item values |

- A subset is syevx: `il`, `iu` 0-based inclusive; `(vl, vu]` half-open. Each item has `k` value
  slots (`iu - il + 1`, or `n` for a value range). Slots past `count` are NaN and their vector
  columns zero. It reads the lower triangle only.
- `svd`: `jobu`/`jobvh` (`"all"`, `"thin"`, `"none"`) override `full_matrices` per side;
  `compute_uv=False` returns `U` and `Vh` as `None`.
- `transpose` and `cond` are real-only (`NotImplementedError` for complex). `cond(a, 2)` is the
  eigenvalue-magnitude ratio of a symmetric `a`, not the general 2-norm condition number.
- Complex `gram` runs `herk`, which needs the vendor BLAS (`NoRouteError` vendor-free).
- The elementwise `add`, `subtract`, `multiply`, `divide`, `axpby` and `scale` are re-exported.

## Known issues

These are library defects; the authoritative list, with causes, is
@ref design_known_defects. The notebooks call them out where they come up.

- **`lanczos` returns a spectrum about three times too wide** (open, cause not located). Use
  `syevx` or the dense solvers for the values.
- **`stedc` without eigenvectors.** The recursive driver may merge from unset leaf vectors under
  `NoEigenVectors` (unverified candidate). The bindings always request vectors internally and
  discard them.
- **Unpreconditioned `syevx` on hard problems.** Can stagnate. Notebook 09 shows the fix.

## Device requirements

The `*_cta` routines map one work-group onto one matrix. They need a sub-group width of 32, so they
are GPU-only. On a CPU they raise `RuntimeError: ... device does not support subgroup size 32 ...`.
They also accept only `n <= 32`. To ask the device directly instead of guessing, call
`syev_variant_support(a, device=...)`.

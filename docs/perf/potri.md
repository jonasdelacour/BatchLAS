# potrs and potri {#perf_potri}

> **Status:** current · routes and tables are hypotheses: nothing here is timed (RTX PRO 6000 sm_120, CUDA 13.2, 2026-10-22)

`potrs` solves with a Cholesky factor, `potri` inverts from one. Both are native compositions over
existing kernels and need no vendor library, so the vendor-free build serves them.

## potrs and potri {#potrs-and-potri}

- **`potrs(A, B, uplo)`** is the second half of `posv`, exposed: `B := A^-1 B` from the factor in the
  `uplo` triangle of `A`, no workspace, no `info`. It has no `Op` and no `BATCHLAS_POTRS_ROUTE`. It runs
  the one-launch fused solve (`getrs_fused.cc`, both substitutions, B in SLM) where
  `ops::potrs::fused_fits` holds (GPU with sub-group size 32, `nrhs <= 8`, `n * nrhs` within the SLM cap),
  and two public `trsm` calls otherwise. `src/ops/potrs/potrs.hh` holds the one implementation: posv's
  `cta` choice is `potrf` + `solve(fused = true)`, its `blocked` choice `potrf` + `solve(fused = false)`.
  `PotrsIsBitIdenticalToPosvOnTheSameRoute` compares the two bit for bit on both families.
- **`potri(A, uplo, full, ws, info)`** follows LAPACK `?potri`: the factor in, the `uplo` triangle of the
  inverse out. `full = true` also writes the conjugate transpose into the other triangle. It forms the
  inverse by solving `A X = I` on an `n x n` identity per item held in the workspace
  (`potri_buffer_size`), then stores `X` into `A`'s triangle (`potri_native.cc`). The cost is a posv with
  `nrhs = n` minus the factorization. cuSOLVER and rocSOLVER have no batched `potri`, so there is no
  vendor family and no `Lib`.
- **`info`** is the `?potri` one: the 1-based index of the first factor diagonal entry that is exactly
  zero. A failed `potrf` is reported by `potrf`'s own `info` (and by `posv`); a non-positive-definite
  item does not disturb the others in any of the three calls. A non-empty `info` shorter than the batch
  throws `invalid_argument`, as the per-item status contract in `docs/cpp-api.md` says (it was silently
  skipped before 2026-10-25). The info kernel's own wrapper (`potri_diag_info`) refuses a short span too,
  with `internal_error`, so its unguarded `out[0, batch)` write never depends on the entry check alone.
- **Diagonal:** the store writes the diagonal's imaginary part as zero, `full` or not: the inverse of a
  Hermitian matrix has a real diagonal, and the solve leaves rounding noise there.
- **Any queue:** `blocked` and `potrs`'s trsm pair run on a CPU queue (NETLIB), the element kernels
  (`potri_native.cc`) have a CPU image; only `fused` needs the GPU. The triangle store is one flat
  grid-strided `nd_range`, like the `src/matrix.cc` element kernels. The earlier 2-D `range(batch, n*n)`
  put the batch on SYCL dimension 0 (CUDA grid y, 65535 work-groups); DPC++ CUDA fitted it at 70001
  items (the restored kernel passed `BatchBeyondTheGridYLimitIsCovered`), but SYCL does not promise it;
  that test is a coverage guard for batch > 65535, not a regression proof for the rewrite.
- **Consumers:** gfn2 Broyden (n = 16), EEQ (n = N + 1), optimize.

## potri selection {#potri-selection}

| family | driver | `can_run` |
|---|---|---|
| `fused` | identity + `potrs_fused_dispatch` | homogeneous batch, GPU, sub-group size 32, `n <= 8` (`nrhs = n` within `getrs_fused`'s 8 and the SLM cap) |
| `blocked` | identity + two public `trsm` | homogeneous batch, `n >= 1` |

Keys `dtype`, `n:log:3`, `batch:log`; there is no `uplo` key (both triangles run the same kernels).
`tuned/potri.sm_89.txt` and `tuned/potri.sm_120.txt` are `transcribed`, untimed: `fused` first for
`n <= 8`, `blocked` first above it (`fused` cannot run there anyway), generated from the ledgers
`benchmarks/results/tuning/ledger/potri.<device>.jsonl` by `scripts/sweep_to_table.py`. The tuner spec
(`tools/tune/potri_spec.cc`) exists, so `batchlas_tune potri` replaces them with timed rows.

> **Warning:** the order `fused` before `blocked` at `n <= 8` is a hypothesis taken from posv's `cta`
> (fused solve) against `blocked` (two trsm) rows; potri at `nrhs = n` was not timed. A native kernel that
> forms the inverse without a workspace identity (triangular inversion then `L^-H L^-1`) is not written; at
> n = 16 to 64 `blocked` pays two trsm launches and an `n^2` workspace write per item, and that is the
> expected place for one to win.

## What the tests pin

`tests/potri_tests.cc`: `A * A^-1 = I` and `A X = B` through `batchlas::verify` (both triangles, an order
ladder through the fused ceiling, padded `ld` and stride, a poisoned unreferenced triangle that must come
back unchanged, real and complex data with nonzero imaginary parts), a bit-for-bit `full` mirror, `info`
with the other items bit-identical to a clean batch, the workspace bound, the fused pin on both sides of
`n = 8`, a 1024-item identical batch bit-identical to item 0 on each pinned family, a 70001-item batch
(past the CUDA grid-y limit) and a short `info` span refused. The tests run on the NETLIB/CPU queue too
(`blocked` only). The deliberate breaks, each with its narrow red set, are named above the tests.

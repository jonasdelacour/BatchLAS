# ortho {#perf_ortho}

> **Status:** current · RTX 4090 (sm_89), CUDA backend, saturating batches unless a section says
> otherwise.

`ortho` (`src/extensions/ortho.cc`) orthogonalizes blocks of vectors for `syevx_lobpcg`,
`syevx_filtered` and `lanczos`, with `k` in the tens. The Cholesky family (`Cholesky`, `Chol2`,
`ShiftChol3`) forms the Gram matrix \f$C = A^H A\f$ with `syrk` for real types at small `k`, and
with GEMM otherwise. The level-3 side of this decision is on
[the level-3 page](level3.md#syrk-for-the-ortho-gram-matrix).

## ortho: the Gram matrix through syrk, per precision

`syrk` computes \f$C = A^H A\f$ at half the arithmetic of a GEMM. The table records the alternatives
that were measured.

| Alternative | Result | Verdict |
| --- | --- | --- |
| `syrk` as a host loop over `cublasXsyrk` (no batched kernel) | 96x slower than GEMM in float; 115 ms against 0.9 ms GEMM in double | rejected |
| `syrk` through `syrk_gram_tiles` (batched Gram tile kernel) | 0.96x to 1.62x over GEMM, by `k` and precision (table below) | taken for `k <= 64` (float) and `k <= 128` (double) |
| `herk` through the Gram tile kernel (complex) | compute bound; GEMM plus Hermitian fold wins at every Gram shape | rejected; complex keeps GEMM |

**`k`.** The single-tile Gram kernel covers `k <= 128`, but the useful limit differs by precision.
End-to-end speed-up of `syrk` over GEMM, with `m = 1024`, batch 512, `Chol2`:

| k | float | double |
|---|---|---|
| 32 | 1.62x | 1.02x |
| 64 | 1.12x | 1.20x |
| 128 | **0.96x** (loss) | 1.34x |

Float at `k = 128` is a wash: the SGEMM it replaces is already at both the compute and the
bandwidth roof. FP64 runs at 1/64 rate on this part, so double is compute bound and the halved
arithmetic lands in full. Hence `gram_max_k` is 64 for float and 128 for double. Above those
limits, double and complex stay on the host loop, which loses to GEMM by 2x at `k = 256`. Absolute
times are on [the level-3 page](level3.md#syrk-for-the-ortho-gram-matrix).

Invariants that the code keeps next to the call:

- `syrk` writes only the lower triangle. `potrf` and `trsm` default to `Uplo::Lower`, and
  `ShiftChol3`'s shift kernel reads only the diagonal.
- `svqb_alg` keeps its GEMM. It scales the whole `k x k` matrix before `syev`, so a one-triangle `C`
  would multiply uninitialised workspace.
- The gate is "`syrk` reaches the Gram tile kernel on this route" (`select::level3_tile_route_available`,
  `src/select/vendor.hh`), not "this is NVIDIA".
- `BATCHLAS_ORTHO_GRAM=gemm` pins GEMM, so the substitution can be measured from one binary.

## ortho: host devices force Householder

On a non-GPU device, `ortho` replaces the requested algorithm with `OrthoAlgorithm::Householder`
(`ortho_force_householder`). The test is on the device, not on `Backend::NETLIB`.

The rule is evaluated in the op and in all three size queries, and they must agree. Sizing for
`Chol2` and running `Householder` would overrun the `BumpAllocator` carved for `tau`, `geqrf_ws` and
`orgqr_ws`.

Any `ortho`-dependent A/B on the NETLIB/CPU build therefore measures Householder, whatever it asks
for. The LOBPCG soft-locking A/B hit exactly this (see [soft locking](syevx.md#lobpcg-soft-locking-by-column-masking)).

## ortho: open debts

- **The transposed CGS arm** builds a view that does not describe the memory; see
  [known defect 1](../design/known-defects.md#defect-1-orthos-transposed-arm-builds-a-view-that-does-not-describe-the-memory).
- The per-precision table has no recorded date or raw file. A re-measure should record both (see
  @ref documentation_conventions).

# ortho {#perf_ortho}

> **Status:** current · RTX 4090 (sm_89), CUDA backend, saturating batches.

`ortho` (`src/ops/ortho/ortho.cc`) orthogonalizes `k`-vector blocks (k in the tens) for
`syevx_lobpcg`, `syevx_filtered` and `lanczos`. The Cholesky family forms \f$C = A^H A\f$ with `syrk`
for real types at small `k`, else GEMM; absolute times are on
[the level-3 page](level3.md#syrk-for-the-ortho-gram-matrix).

## ortho: the Gram matrix through syrk, per precision

`gram_max_k` is 64 for float and 128 for double (`ortho.cc:175`). End-to-end `syrk` over GEMM,
`m = 1024`, batch 512, `Chol2` (no recorded date or raw file):

| k | float | double |
|---|---|---|
| 32 | 1.62x | 1.02x |
| 64 | 1.12x | 1.20x |
| 128 | **0.96x** | 1.34x |

Float SGEMM at `k = 128` is already at the roof; FP64 at 1/64 rate is compute bound, so halved
arithmetic lands in full.

| Alternative | Result | Verdict |
| --- | --- | --- |
| host loop over `cublasXsyrk` | 96x slower than GEMM (float); 115 ms vs 0.9 ms (double) | rejected |
| complex `herk` via the Gram tile kernel | GEMM + Hermitian fold wins at every shape | complex keeps GEMM |

- `syrk` writes only the lower triangle; `potrf`/`trsm` default to `Uplo::Lower` and `ShiftChol3`
  reads only the diagonal.
- `svqb_alg` keeps GEMM: it scales the full `k x k` matrix before `syev`.
- The gate is `select::level3_tile_substitution_enabled` (`src/select/vendor.hh:105`), not "this is
  NVIDIA". `BATCHLAS_ORTHO_GRAM=gemm` pins GEMM for A/B from one binary.

## ortho: host devices force Householder

On a non-GPU device `ortho_force_householder` replaces the requested algorithm with Householder (a
device test, not `Backend::NETLIB`). The op and all three size queries apply it, and must agree, or
`Chol2` sizing would overrun the `tau`/`geqrf_ws`/`orgqr_ws` carve. Any `ortho`-dependent CPU A/B
therefore measures Householder (see [soft locking](syevx.md#lobpcg-soft-locking-by-column-masking)).

## ortho: open debts

- The transposed CGS arm ([known defect 1](../design/known-defects.md#defect-1-orthos-transposed-arm-builds-a-view-that-does-not-describe-the-memory))
  is fixed but untimed.
- The per-precision table needs a dated re-measure with a raw file (@ref documentation_conventions).

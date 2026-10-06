# ortho: the Gram product and the algorithm rules {#perf_ortho}

> **Covers:** batched orthogonalization `ortho` (`src/extensions/ortho.cc`): how the Gram matrix
> \f$C = A^H A\f$ is formed, per precision, and the rule that forces Householder on host devices.
> **Status:** current. Moved out of the `ortho.cc` comments on 2026-09-30.
> **Machine:** RTX 4090 (sm_89), CUDA backend, saturating batches, unless a section says otherwise.
> **Measured:** the syrk-vs-GEMM re-measure after `syrk_gram_tiles` landed; the source comment
> does not record the date.

`ortho` is called by `syevx_lobpcg`, `syevx_filtered` and `lanczos` with `k` (the number of vectors)
a block size in the tens. The Cholesky family (`Cholesky`, `Chol2`, `ShiftChol3`) forms the Gram
matrix with `syrk` for real types at small `k`, and with a GEMM otherwise. The level-3 side of the
same decision, including the original 70x to 100x regression, is on
[the level-3 page](level3.md#syrk-for-the-ortho-gram-matrix).

## ortho: the Gram matrix through syrk, per precision

`C = A^H A` is exactly what `syrk` spells, at half the arithmetic of a GEMM. PR #61 measured the
substitution and **rejected** it, correctly at the time: `syrk` reached no batched kernel at these
shapes and fell to a host loop over `cublasXsyrk`, 96x slower in float and 115 ms against a 0.9 ms
GEMM in double. `syrk_gram_tiles` is that missing kernel, and with it the substitution was
re-measured and taken, under two conditions.

**`k`.** The single-tile Gram kernel covers `k <= 128`, but the useful limit differs by precision.
End to end, `m = 1024`, batch 512, `Chol2`, speed-up of syrk over the GEMM:

| k | float | double |
|---|---|---|
| 32 | 1.62x | 1.02x |
| 64 | 1.12x | 1.20x |
| 128 | **0.96x** (loss) | 1.34x |

Float at `k = 128` is a wash because the SGEMM it replaces is already at both the compute and the
bandwidth roof, so the halved arithmetic buys nothing. FP64 runs at 1/64 rate on this part, so
double is squarely compute bound, the halving lands in full, and the gain grows with `k` where
float's shrinks. Hence `gram_max_k` is 64 for float and 128 for double, not one number for both.
Above those limits double and complex are still on the host loop, which loses to the GEMM by 2x at
`k = 256`. The absolute times behind the float `k = 32` and `k = 128` cells and the double
`k = 128` cell are on [the level-3 page](level3.md#syrk-for-the-ortho-gram-matrix).

**Real types only.** A complex multiply is four real ones, so `herk` is compute bound where `syrk`
is bandwidth bound, and the existing GEMM plus Hermitian fold beats the tile kernel at every Gram
shape (see [herk on the gram tile kernel](level3.md#herk-on-the-gram-tile-kernel)). Complex keeps
the GEMM.

Invariants that the code keeps next to the call:

- `syrk` writes only the lower triangle. Everything downstream of the two Gram call sites reads
  exactly that: `potrf` and `trsm` default to `Uplo::Lower`, and `ShiftChol3`'s shift kernel reads
  only the diagonal.
- `svqb_alg` keeps its GEMM. It scales the whole `k x k` before `syev`, so a one-triangle `C` would
  multiply uninitialised workspace.
- The gate is "`syrk` reaches the Gram tile kernel on this route"
  (`select::level3_tile_route_available`, `src/select/vendor.hh`), not "this is NVIDIA".
- `BATCHLAS_ORTHO_GRAM=gemm` pins the GEMM, so the substitution stays measurable from one binary
  rather than needing a build of the parent commit.

## ortho: host devices force Householder

On a non-GPU device `ortho` replaces the requested algorithm by `OrthoAlgorithm::Householder`
(`ortho_force_householder`). The question is asked about the **device**, not the backend enum:
`B == Backend::NETLIB` was the old spelling, and it stands for "there are no device kernels here",
which is a property of the device and stays correct for a host queue reached through any backend.

The rule is asked in one place, by the op **and** by all three size queries. They must agree: sized
as `Chol2` and then run as `Householder`, `ortho_layout` would carve `tau`, `geqrf_ws` and
`orgqr_ws` out of a `BumpAllocator` that was never sized for them, a workspace overrun. The outcome
did not change when the rule was centralised (NETLIB is the only backend on a host device), so it
is a by-construction guarantee rather than a bug fix.

A consequence for measurements: any `ortho`-dependent A/B run on the NETLIB/CPU build measures
Householder, whatever it asks for. The LOBPCG soft-locking A/B hit exactly this (see
[soft locking](syevx.md#lobpcg-soft-locking-by-column-masking)).

## ortho: open debts

- **The transposed CGS arm** builds a view that does not describe the memory; see
  [known defect 1](../design/known-defects.md#defect-1-orthos-transposed-arm-builds-a-view-that-does-not-describe-the-memory).
- **No date or raw file** is recorded for the per-precision table above; a re-measure should
  record both (see @ref documentation_conventions).

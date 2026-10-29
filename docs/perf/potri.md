# potrs and potri {#perf_potri}

> **Status:** current · potri timed and tuned on RTX 4090 (sm_89), CUDA 13.2, 2026-10-26; potrs untimed

`potrs` solves with a Cholesky factor, `potri` inverts from one. Both are native compositions over
existing kernels and need no vendor library, so the vendor-free build serves them.

## potrs and potri {#potrs-and-potri}

- **`potrs(A, B, uplo)`** is `posv`'s second half: `B := A^-1 B` from the factor, no workspace, no
  `info`. A routed op ([potrs selection](potrs.md#potrs-selection)): the one-launch fused solve or two
  public `trsm`. `src/ops/potrs/kernels/potrs_trsm.cc` is the one implementation: posv's `cta` is
  `potrf` + `solve(fused = true)`, `blocked` is `potrf` + `solve(fused = false)`;
  `FusedMatchesPosvCtaAndTrsmMatchesPosvBlockedBitForBit` checks it bit for bit.
- **`potri(A, uplo, full, ws, info)`** follows LAPACK `?potri`: `?trtri` then `?lauum`, in place, only the
  `uplo` triangle read or written; `full = true` also writes the conjugate transpose. cuSOLVER and
  rocSOLVER have no batched `potri`, so there is no vendor family. The previous identity solve
  (`A X = I`, ~3x the work, `n^2 * batch` workspace) lost by 2.75x (float) and 2.97x (cfloat) geomean
  (sm_89, 2026-10-26, 78 cells, raw data in the job's `perf-potri/ab/`) and was removed.
- **`info`** is `?potri`'s: the 1-based index of the first exactly zero factor diagonal entry. Such an
  item is left unchanged on both families, with or without `info`; other items are undisturbed. A
  non-empty `info` shorter than the batch throws `invalid_argument`; the writing kernels also refuse a
  short span with `internal_error`.
- **Diagonal:** the imaginary part of the diagonal is written as zero, `full` or not.
- **Any queue:** both families and the trsm pair run on a CPU queue (NETLIB: blocked's children go to
  CBLAS). Every potri launch is one-dimensional, so the batch never lands on CUDA grid y or z
  (`BatchBeyondTheGridYLimitIsCovered`, 70001 items).
- **Consumers:** gfn2 Broyden (n = 16), EEQ (n = N + 1), optimize.

## potri selection {#potri-selection}

| family | driver | `can_run` (`src/ops/potri/potri.cc`) |
|---|---|---|
| `cta` | `sycl_potri::potri_cta`, mode `Potri`: trtri + lauum in one launch | homogeneous, `n <= cta_max_n<T>(d.slm_budget, d.max_wg)` |
| `blocked:nb=64`, `blocked:nb=128` (float only) | `ops::potri::blocked::run` | homogeneous, `min(nb, n) <= cta_max_n` |

`cta_max_n` (`potri_native.hh`) is the largest order whose padded tile fits local memory and whose
register tiles fit two per work-item at 512 work-items. sm_89: float 152, double and cfloat 108,
cdouble 76; the OpenCL CPU device admits 176 for float. With no table the last-resort order is
`blocked`, then `cta`. Keys `dtype`, `n:log:3`, `batch:log`; no `uplo` key (Upper runs the same
kernels on `L = U^H`). nb = 32 won no cell and is not a candidate.

**Tables.** `tuned/potri.sm_89.txt` float and cfloat are `deep` (2026-10-26, RTX 4090 GPU 1, 177
cells). cta wins every cell it can run; blocked:nb=64 above its ceiling; float nb=128 from n ~ 330.
double and cdouble, cells over the 8 GiB operand cap, and all of `tuned/potri.sm_120.txt` are
`transcribed` from those rankings (run lines `transcribed-f517330f`, `-gap`). Bracketing cells, batch
2048, ms (runner-up in brackets):

| dtype | n | winner |
|---|---:|---|
| float | 149 | `cta` 0.9035 (nb=64 1.463) |
| float | 157 | `nb=64` 1.465 (nb=128 1.987), past the cta ceiling |
| float | 256 | `nb=64` 3.421 (nb=128 3.464) |
| float | 330 | `nb=128` 6.998 (nb=64 7.736) |
| cfloat | 96 | `cta` 0.7603 (nb=64 1.176) |
| cfloat | 128 | `nb=64` 1.785, past the cta ceiling |

## potri: the algorithm {#potri-the-algorithm}

Upper is Lower on `L = U^H` (`A^-1 = X^H X`, `X = L^-1`); kernels load `conj` of the transposed
position.

**cta** (`kernels/potri_cta.cc`, one work-group per item): the block sits in a tile padded to TS (4;
2 for cdouble) with an odd leading dimension; strict lower holds `M = L diag(L)^-1`, upper holds
`P = diag(L) L^-1` transposed, so the solve never overwrites what it reads. trti2 is right-looking by
TS pivots: one barrier per tile row, then a rank-TS update per tile (2 TS^2 local loads per TS^3
multiply-adds; a per-pivot version was instruction-bound). lauu2 then accumulates `X^H X` in the same
register tiles and stores coalesced, `full` mirroring in the same pass. A zero-diagonal item returns
early, unchanged. Modes `Trtri` (every nb x nb diagonal block) and `Lauum` (one block) serve blocked.

**blocked:nb** (`kernels/potri_blocked.cc`), LAPACK's blocked loops on public ops (each child takes its
own table):

- trtri: one `Trtri` launch for all diagonal blocks, then per block column two trmm.
- lauum: trmm + gemm into `W`, copy back, `Lauum` on the diagonal block, then herk (complex) / syrk
  (real).
- `W` is one `n x nb` panel per item; the batch runs in parts (`tuning::blocked_chunk_items`, potrf's
  rule), so the cdouble corner (n = 512, batch 32768) needs 64 MiB, not the identity solve's 137 GB. At
  `n <= nb` it is one cta `Potri` launch and no workspace. `potri_finish` zeroes the diagonal's
  imaginary part and mirrors for `full`.
- A failed item stays unchanged without a host round trip (`sycl_potri::Stash`): the `Trtri` launch
  writes `info` (or workspace flags) and stashes the item's triangle; the last `Lauum` launch restores
  it, and `potri_finish` skips it. Healthy-batch cost is unchanged (new/base 0.992-1.002, nine cells,
  sm_89, 2026-10-26).

## potri measured against the identity solve {#potri-measured-against-the-identity-solve}

sm_89, 2026-10-26, `batchlas_tune potri` custom protocol (cold L2, arms alternated): float 40 cells
geomean 2.75x (1.43x-5.19x), cfloat 38 cells 2.97x (one tie, 0.96x at n = 6, b = 8192). The old fused
family (n <= 8) lost to cta by 1.6x-4.0x everywhere. Raw JSONL: the job's `perf-potri/ab/`.

## potri: what the tests pin {#potri-what-the-tests-pin}

`tests/potri_tests.cc`: `A * A^-1 = I` through `batchlas::verify` on Auto and each pinned candidate,
both triangles, orders through the tile width and both block orders up to n = 300; graded spectra
(1e3 float/cfloat, 1e6 double/cdouble); padded `ld`/stride with a poisoned unreferenced triangle;
complex data; the bit-for-bit `full` mirror; `info` with the failed item unchanged and the rest
bit-identical on every family; workspace per family; a forced three-item chunk; the cta pin at its
ceiling (graded) and refused one past it; 1024 identical items bit-identical (n = 37, 100); a
70001-item batch; a short `info` span refused; and `reached` coverage rows (blocked's children exactly
trmm, gemm, herk/syrk; cta's none).

Deliberate breaks (2026-10-26, re-run 2026-10-27 on `tests/test_support/` inputs; CUDA and NETLIB,
four types, 112 cases): each break below turned only its listed tests red.

| break | red |
|---|---|
| cta trti2 skips its last sub-diagonal term | Beyond, Full, Graded, Inverts, Parts, Saturating, Straddle |
| cta Upper loads `U^T`, or store mirrors without `conj` | Beyond, Full, Graded, Inverts, Parts (complex only) |
| cta per-tile-row barrier removed | Full, Graded, Info (CUDA), Inverts, Parts, Saturating, Straddle |
| cta `info` 0-based | Info |
| blocked trtri's `-W X11` trmm replaced by a copy | Full, Graded, Inverts, Parts, Saturating (CUDA) |
| blocked lauum gemm leg, or herk/syrk, dropped | Coverage, Graded, Inverts, Full (dz) |
| blocked parts all start at item 0 | Parts |
| last `Lauum` never restores the stash; `potri_finish` ignores flags | Info |

Diagonally dominant inputs missed a broken substitution at the float cta ceiling and a dropped lauum
gemm; graded inputs catch both, so the ceiling launch uses a graded matrix.

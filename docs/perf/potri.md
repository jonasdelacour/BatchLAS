# potrs and potri {#perf_potri}

> **Status:** current · potri timed and tuned on RTX 4090 (sm_89), CUDA 13.2, 2026-10-26; potrs untimed

`potrs` solves with a Cholesky factor, `potri` inverts from one. Both are native compositions over
existing kernels and need no vendor library, so the vendor-free build serves them.

## potrs and potri {#potrs-and-potri}

- **`potrs(A, B, uplo)`** is the second half of `posv`, exposed: `B := A^-1 B` from the factor in the
  `uplo` triangle of `A`, no workspace, no `info`. Since 2026-10-26 it is a routed op (`Op::potrs`,
  `BATCHLAS_POTRS_ROUTE`, measured tables): the one-launch fused solve (`getrs_fused.cc`, both
  substitutions, B in SLM) or two public `trsm` calls, chosen per cell
  ([potrs selection](potrs.md#potrs-selection)). `src/ops/potrs/potrs.hh` holds the one implementation:
  posv's `cta` choice is `potrf` + `solve(fused = true)`, its `blocked` choice `potrf` + `solve(fused = false)`.
  `FusedMatchesPosvCtaAndTrsmMatchesPosvBlockedBitForBit` (`tests/potrs_tests.cc`) compares them bit for bit.
- **`potri(A, uplo, full, ws, info)`** follows LAPACK `?potri`: the factor in, the `uplo` triangle of the
  inverse out, by LAPACK's algorithm, `?trtri` (the factor's triangular inverse, in place) then `?lauum`
  (`X^H X` for Lower, `X X^H` for Upper, in place): n^3 / 3 + n^3 / 3 multiply-adds per item and only the
  `uplo` triangle read or written ([the algorithm](#potri-the-algorithm)). `full = true` also writes the
  conjugate transpose into the other triangle. cuSOLVER and rocSOLVER have no batched `potri`, so there is
  no vendor family and no `Lib`. Until 2026-10-26 potri solved `A X = I` on an `n x n` identity per item in
  the workspace (two full `trsm`, about 3x LAPACK's work, `n^2 * batch` workspace, 137 GB at the cdouble
  grid corner); [the A/B](#potri-measured-against-the-identity-solve) retired it.
- **`info`** is the `?potri` one: the 1-based index of the first factor diagonal entry that is exactly
  zero. Such an item is left unchanged, as `?trtri` returns before writing, on both families and with
  or without `info` (blocked wrote inf/NaN into it before 2026-10-26). A failed `potrf` is reported by
  `potrf`'s own `info` (and by `posv`); a non-positive-definite item does not disturb the others in any
  of the three calls. A non-empty `info` shorter than the batch throws `invalid_argument`, as the
  per-item status contract in `docs/cpp-api.md` says (it was silently skipped before 2026-10-25). The
  kernels that write it (`potri_cta`, and the blocked driver for its `Trtri` launch) refuse a short span
  too, with `internal_error`, so their unguarded `[0, batch)` writes never depend on the entry check alone.
- **Diagonal:** potri writes the diagonal's imaginary part as zero, `full` or not: the inverse of a
  Hermitian matrix has a real diagonal (cta stores lauu2's real sum; blocked zeroes it in `potri_finish`).
- **Any queue:** both potri families and `potrs`'s trsm pair run on a CPU queue (NETLIB): the potri
  kernels (`potri_cta.cc`, `potri_native.cc`) need no sub-group size and have a CPU image, blocked's
  children go to CBLAS there. Every potri launch is one-dimensional (one work-group per item or per
  diagonal block, or grid-strided), so the batch never lands on CUDA grid y or z;
  `BatchBeyondTheGridYLimitIsCovered` (70001 items) is a coverage guard for that, per family.
- **Consumers:** gfn2 Broyden (n = 16), EEQ (n = N + 1), optimize.

## potri selection {#potri-selection}

| family | driver | `can_run` (`src/ops/potri/potri.cc`) |
|---|---|---|
| `cta` | `sycl_potri::potri_cta`, mode `Potri`: trtri + lauum in one launch | homogeneous batch, `n <= cta_max_n<T>(d.slm_budget, d.max_wg)` |
| `blocked:nb=64`, `blocked:nb=128` (float only) | `ops::potri::blocked::run` (`potri_blocked.hh`) | homogeneous batch, `min(nb, n) <= cta_max_n` (one diagonal block must fit the CTA kernel) |

`cta_max_n` (`potri_native.hh`) is the largest order whose padded tile (`(mp + 1) * mp` scalars plus `mp`
reals, `mp` = n rounded up to the tile width TS, the 48 KB launch hole padded) fits the local memory and
whose `T (T + 1) / 2` register tiles fit two per work-item at 512 work-items. On sm_89 (99 KB less the
4 KB reserve): float 152, double and cfloat 108, cdouble 76; the host device (OpenCL CPU) admits 176 for float. Neither family needs a sub-group size, so both run on the host
queue too (NETLIB: the children go to CBLAS, the CTA kernel to the CPU device). With no table (the CPU)
the last-resort order is `blocked`, then `cta`.

Keys `dtype`, `n:log:3`, `batch:log`; there is no `uplo` key (Upper runs the same kernels on `L = U^H`).
nb = 32 was a candidate in the A/B and won no cell; nb = 128 exists for float only, since a 128-order CTA
block fits no other type's local memory. 

**The tables.** `tuned/potri.sm_89.txt` float and cfloat rows are `deep` (`batchlas_tune potri --tier
deep`, lean-cold, 2026-10-26, RTX 4090 GPU 1 headless under the box-wide timing lock, no foreign process
seen during any of the ten chunks; 177 cells with refinement, which bracketed the cta ceiling at
n = 142/149/157 and the float nb crossover at n = 314/330/347). cta wins every cell it can run, up to its
ceiling; blocked:nb=64 above it, and for float nb=128 from n ~ 330 (and at n = 256, b = 128). double and
cdouble are `transcribed` from float's and cfloat's measured ranking (FP64 is not tuned on a 1:64 part,
tuned/README.md), each filtered to the families the type can run (double's cta ceiling is 108, cdouble's
76); the float and cfloat cells over the campaign's 8 GiB operand cap are transcribed from their nearest
measured cell, and every `tuned/potri.sm_120.txt` row from the sm_89 ranking (sm_120 has the same 99 KB of
local memory; not timed there). Run lines `transcribed-f517330f` and `transcribed-f517330f-gap` in the
ledgers name the rule. Measured winners, ms (runner-up in brackets):

| dtype | n | batch 128 | batch 2048 | batch 8192 |
|---|---:|---|---|---|
| float | 4 | `cta` 0.008647 (nb=64 0.01314) | `cta` 0.01020 (nb=128 0.01570) | `cta` 0.01917 (nb=64 0.02984) |
| float | 8 | `cta` 0.01030 (nb=128 0.01422) | `cta` 0.01172 (nb=128 0.01717) | `cta` 0.02460 (nb=128 0.03534) |
| float | 16 | `cta` 0.01203 (nb=64 0.01801) | `cta` 0.01630 (nb=64 0.02320) | `cta` 0.04108 (nb=128 0.05812) |
| float | 32 | `cta` 0.01602 (nb=64 0.02329) | `cta` 0.03789 (nb=64 0.05209) | `cta` 0.1215 (nb=128 0.1705) |
| float | 64 | `cta` 0.02246 (nb=128 0.03088) | `cta` 0.1207 (nb=64 0.1653) | `cta` 0.4351 (nb=64 0.6892) |
| float | 96 | `cta` 0.03092 (nb=128 0.04051) | `cta` 0.3893 (nb=128 0.4998) | `cta` 1.601 (nb=64 2.190) |
| float | 128 | `cta` 0.04729 (nb=128 0.05924) | `cta` 0.7062 (nb=64 0.8231) | `cta` 2.632 (nb=64 3.097) |
| float | 142 | `cta` 0.05726 (nb=64 0.1342) | `cta` 0.8082 (nb=64 1.349) | `cta` 3.236 (nb=64 5.163) |
| float | 149 | `cta` 0.06293 (nb=64 0.1454) | `cta` 0.9035 (nb=64 1.463) | `cta` 3.621 (nb=64 5.588) |
| float | 157 | `nb=64` 0.1519 (nb=128 0.1731) | `nb=64` 1.465 (nb=128 1.987) | `nb=64` 5.927 (nb=128 7.929) |
| float | 192 | `nb=64` 0.1712 (nb=128 0.1871) | `nb=64` 1.872 (nb=128 2.390) | `nb=64` 7.535 (nb=128 9.463) |
| float | 256 | `nb=128` 0.2285 (nb=64 0.2853) | `nb=64` 3.421 (nb=128 3.464) | - |
| float | 330 | - | `nb=128` 6.998 (nb=64 7.736) | - |
| float | 347 | - | `nb=128` 7.678 (nb=64 8.333) | - |
| float | 384 | `nb=128` 0.5154 (nb=64 0.6460) | `nb=128` 8.508 (nb=64 9.452) | - |
| float | 512 | `nb=128` 0.9515 (nb=64 1.238) | - | - |
| cfloat | 4 | `cta` 0.009133 (nb=64 0.01355) | `cta` 0.01318 (nb=64 0.01883) | `cta` 0.02564 (nb=64 0.03672) |
| cfloat | 8 | `cta` 0.01030 (nb=64 0.01566) | `cta` 0.01712 (nb=64 0.02314) | `cta` 0.03847 (nb=64 0.05065) |
| cfloat | 16 | `cta` 0.01418 (nb=64 0.01991) | `cta` 0.02797 (nb=64 0.03617) | `cta` 0.07063 (nb=64 0.09168) |
| cfloat | 32 | `cta` 0.02011 (nb=64 0.02734) | `cta` 0.07283 (nb=64 0.08856) | `cta` 0.2352 (nb=64 0.2982) |
| cfloat | 64 | `cta` 0.03196 (nb=64 0.04090) | `cta` 0.2647 (nb=64 0.3227) | `cta` 0.9926 (nb=64 1.411) |
| cfloat | 96 | `cta` 0.05412 (nb=64 0.1148) | `cta` 0.7603 (nb=64 1.176) | `cta` 3.073 (nb=64 4.480) |
| cfloat | 128 | `nb=64` 0.1520 | `nb=64` 1.785 | `nb=64` 7.164 |
| cfloat | 192 | `nb=64` 0.3612 | `nb=64` 4.917 | - |
| cfloat | 256 | `nb=64` 0.6638 | `nb=64` 9.331 | - |
| cfloat | 384 | `nb=64` 1.749 | - | - |
| cfloat | 512 | `nb=64` 3.551 | - | - |

## potri: the algorithm {#potri-the-algorithm}

LAPACK's `?potri` is `?trtri` then `?lauum`. Upper is Lower on `L = U^H`: `A = U^H U = L L^H`, so
`A^-1 = X^H X` with `X = L^-1`; every kernel loads `conj` of the transposed position and stores back the
same way.

**cta** (`kernels/potri_cta.cc`, one work-group per item): the order-m block is staged into a tile padded
to a multiple of TS (4; 2 for cdouble) with an odd leading dimension. Its strict lower triangle holds
`M = L diag(L)^-1` (the multipliers), the diagonal 1 and the upper triangle `P = diag(L) L^-1`
transposed, so the solve never overwrites what it still reads. trti2 is right-looking by TS pivots: the
tile row of block K substitutes within its TS rows in registers and publishes them, one barrier, then every
tile below takes the rank-TS update `acc -= M(rows, K) P(K, cols)` (2 TS^2 local loads per TS^3
multiply-adds, one barrier per tile row instead of one per pivot; the per-pivot first version ran the
n = 64 trtri at 0.9 us per pivot, instruction-bound on predicates). Then `X = diag(L)^-1 P` goes to the
upper triangle as `X^T`, the strict lower one is zeroed, and lauu2 accumulates `Y = X^H X` in the same
register tiles (`Y(i, j) = sum_k conj(X(k, i)) X(k, j)`, zero below the diagonal of X for free). Y is
staged in the free strict lower triangle and stored coalesced; `full` writes its conjugate into the other
triangle in the same pass, the diagonal real. Each work-item owns one or two lower-triangle tiles. An item
whose factor has an exactly zero diagonal entry gets its `info` and is left unchanged (a work-group
uniform early return). The same kernel has two other modes for the blocked driver: `Trtri` inverts every
nb x nb diagonal block of the item (one work-group per block) and `Lauum` forms `X^H X` of one block.

**blocked:nb** (`potri_blocked.hh`), LAPACK's blocked loops on public ops, so each child takes its own table:

- trtri: all diagonal blocks inverted by one `Trtri` launch; then per block column (Lower: last first)
  `W = X22 L21` (trmm, Left) and `X21 = -W X11` (trmm, Right, out of place back into A); Upper mirrors it.
- lauum (Lower, ascending block rows): `W = Xii^H X(i, 0:i)` (trmm), `W += X(i+ib:, i)^H X(i+ib:, 0:i)`
  (gemm, beta 1), copy W back, `Lauum` on the diagonal block, `Aii += X(i+ib:, i)^H X(i+ib:, i)` (herk for
  complex, syrk for real; real types pass Trans, complex ConjTrans).
- trmm writes out of place, so W holds one off-diagonal panel, `n x nb` per item. The batch runs in parts
  (`tuning::blocked_chunk_items` on the triangle plus W, potrf's rule: 75 % of L2, a multiple of the CU
  count), so W and the pointer arrays (three, one per view of a call) are one part's: at the cdouble grid
  corner (n = 512, batch 32768) the part is 128 items and W 64 MiB, where the identity solve asked for
  137 GB. At `n <= nb` it is the cta family's one `Potri` launch and no workspace. `potri_finish`
  zeroes the diagonal's imaginary part (herk's is the child's business) and mirrors for `full`.
- A failed item (an exactly zero diagonal entry) stays unchanged without a host round trip
  (`sycl_potri::Stash`): every work-group of the `Trtri` launch scans its item's whole diagonal, writes
  `info` (or a batch of flags in the workspace when no `info` is passed) and, for a failed item, copies
  its block column's triangle to the item's slot and returns; the trmm, gemm and herk/syrk children then
  write garbage into that item, and the part's last launch, `Lauum` on the last diagonal block, copies
  the slot back instead. `potri_finish` skips flagged items. The slots are one packed `uplo` triangle per
  item of a part (256 MiB at the cdouble corner above), the price of leaving the item alone, plus one
  int per item of the batch. No launch is added, and a healthy batch's cost is unchanged: an A/B of the
  pinned blocked arm against the previous build (sm_89, 2026-10-26, 14 alternated passes, the first
  dropped) gave new/base 0.992-1.002 at nine float and complex<float> cells, n = 128-512, batch 128 and
  2048. At `n <= nb` the one `Potri` launch that replaced `Trtri`, `Lauum` and the scan takes 0.78x the time (float
  n = 96, nb = 128) and 0.82x (complex<float> n = 48, nb = 64) at batch 2048. Before 2026-10-26 `info` came from a separate scan launch, `potri_diag_info` (one 32-wide work-group per item; the earlier
  one-thread-per-item scan cost 27 us at batch 1024 for n = 64, more than the trmm it preceded).

## potri measured against the identity solve {#potri-measured-against-the-identity-solve}

RTX 4090 (sm_89, GPU 1, headless), CUDA 13.2, DPC++ `/opt/dpcpp-cuda`, 2026-10-26, `batchlas_tune potri`
custom protocol (`--reps 8 --warm 0.5 --passes 2`, cold L2, every arm verified, the arms alternated),
under the box-wide timing lock; medians in ms over all reps of both passes (16 to 32 per cell). Every
arm in one binary: `cta`, `blocked:nb` and the old `fused` (identity + the one-launch potrs kernel,
n <= 8) and `identity` (identity + two public trsm, the old `blocked`). old/new = best old arm over best
new arm. Cells above 4 GiB of operands were skipped (the old families' n^2 workspace).

- float, 40 cells: geomean **2.75x**, range 1.43x (n = 6, b = 8192) to 5.19x (n = 96, b = 128).
- cfloat, 38 cells: geomean **2.97x**, range 0.96x (n = 6, b = 8192: identity 0.0394 against cta 0.0412,
  a tie at the noise level) to 5.58x (n = 48, b = 8192). The cfloat n >= 64 run saw another agent's
  `tuned_tables_tests` start on the same GPU (logged by the run's monitor); its arms were alternated.
- The fused family lost every cell it could run (n <= 8) to cta by 1.6x to 4.0x and to identity at
  n >= 6, b = 8192: the hypothesis "fused first at n <= 8" in the transcribed tables was wrong.
- The crossover from cta to blocked is in order: cta wins every cell it can run in this sweep (float to
  n = 128, cfloat to n = 96); nb = 128 beats nb = 64 from n = 256 for float.

A subset (all 78 cells, the raw JSONL and the summary script are in the job's evidence directory,
`perf-potri/ab/`):

| dtype | n | batch | cta | blocked:nb=64 | blocked:nb=128 | fused | identity | old/new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cfloat | 4 | 128 | 0.0096 | 0.0142 | - | 0.0224 | 0.0263 | **2.34** |
| cfloat | 4 | 1024 | 0.0110 | 0.0173 | - | 0.0267 | 0.0306 | **2.43** |
| cfloat | 4 | 8192 | 0.0312 | 0.0445 | - | 0.0682 | 0.0651 | **2.09** |
| cfloat | 8 | 128 | 0.0108 | 0.0154 | - | 0.0346 | 0.0238 | **2.21** |
| cfloat | 8 | 1024 | 0.0123 | 0.0172 | - | 0.0414 | 0.0273 | **2.22** |
| cfloat | 8 | 8192 | 0.0386 | 0.0506 | - | 0.1558 | 0.0429 | **1.11** |
| cfloat | 16 | 128 | 0.0144 | 0.0202 | - | - | 0.0300 | **2.09** |
| cfloat | 16 | 1024 | 0.0172 | 0.0232 | - | - | 0.0474 | **2.75** |
| cfloat | 16 | 8192 | 0.0710 | 0.0911 | - | - | 0.1446 | **2.04** |
| cfloat | 32 | 128 | 0.0208 | 0.0285 | - | - | 0.0734 | **3.53** |
| cfloat | 32 | 1024 | 0.0476 | 0.0590 | - | - | 0.1854 | **3.89** |
| cfloat | 32 | 8192 | 0.2368 | 0.3000 | - | - | 1.2336 | **5.21** |
| cfloat | 64 | 128 | 0.0324 | 0.0417 | - | - | 0.1648 | **5.08** |
| cfloat | 64 | 1024 | 0.1385 | 0.1677 | - | - | 0.6444 | **4.65** |
| cfloat | 64 | 8192 | 0.9909 | 1.4091 | - | - | 5.4945 | **5.55** |
| cfloat | 128 | 128 | - | 0.1532 | - | - | 0.4540 | **2.96** |
| cfloat | 128 | 1024 | - | 0.8882 | - | - | 3.3916 | **3.82** |
| cfloat | 256 | 128 | - | 0.6590 | - | - | 1.7408 | **2.64** |
| cfloat | 256 | 1024 | - | 4.6618 | - | - | 15.7109 | **3.37** |
| cfloat | 512 | 128 | - | 3.5667 | - | - | 8.7443 | **2.45** |
| float | 4 | 128 | 0.0116 | 0.0189 | 0.0192 | 0.0279 | 0.0369 | **2.41** |
| float | 4 | 1024 | 0.0094 | 0.0146 | 0.0148 | 0.0195 | 0.0242 | **2.07** |
| float | 4 | 8192 | 0.0222 | 0.0326 | 0.0329 | 0.0448 | 0.0488 | **2.02** |
| float | 8 | 128 | 0.0108 | 0.0168 | 0.0168 | 0.0282 | 0.0303 | **2.61** |
| float | 8 | 1024 | 0.0137 | 0.0210 | 0.0208 | 0.0403 | 0.0417 | **2.95** |
| float | 8 | 8192 | 0.0249 | 0.0357 | 0.0356 | 0.0880 | 0.0387 | **1.55** |
| float | 16 | 128 | 0.0135 | 0.0203 | 0.0198 | - | 0.0372 | **2.76** |
| float | 16 | 1024 | 0.0170 | 0.0233 | 0.0239 | - | 0.0454 | **2.67** |
| float | 16 | 8192 | 0.0418 | 0.0589 | 0.0587 | - | 0.0743 | **1.78** |
| float | 32 | 128 | 0.0176 | 0.0259 | 0.0259 | - | 0.0578 | **3.28** |
| float | 32 | 1024 | 0.0233 | 0.0326 | 0.0319 | - | 0.0668 | **2.86** |
| float | 32 | 8192 | 0.1220 | 0.1714 | 0.1706 | - | 0.2435 | **1.99** |
| float | 64 | 128 | 0.0220 | 0.0308 | 0.0308 | - | 0.1032 | **4.70** |
| float | 64 | 1024 | 0.0668 | 0.0895 | 0.0893 | - | 0.1871 | **2.80** |
| float | 64 | 8192 | 0.4348 | 0.6887 | 0.6890 | - | 1.7092 | **3.93** |
| float | 128 | 128 | 0.0476 | 0.0874 | 0.0595 | - | 0.2245 | **4.71** |
| float | 128 | 1024 | 0.3279 | 0.3841 | 0.4067 | - | 0.7663 | **2.34** |
| float | 128 | 8192 | 2.6357 | 3.1050 | 3.4973 | - | 8.1574 | **3.09** |
| float | 256 | 128 | - | 0.2861 | 0.2292 | - | 0.5492 | **2.40** |
| float | 256 | 1024 | - | 1.7812 | 1.7256 | - | 4.9810 | **2.89** |
| float | 512 | 128 | - | 1.2768 | 0.9781 | - | 2.7557 | **2.82** |

## potri: what the tests pin {#potri-what-the-tests-pin}

`tests/potri_tests.cc` (potri section): `A * A^-1 = I` through `batchlas::verify` on Auto and each
compiled candidate pinned, both triangles, an order ladder through the tile width, both block orders and
their ragged neighbours up to n = 300 (lauum's gemm leg needs three blocks); graded spectra (condition
1e3 float/cfloat, 1e6 double/cdouble, random unitary eigenvectors); padded `ld` and stride with a poisoned
unreferenced triangle that must come back unchanged; complex data with nonzero imaginary parts; the
bit-for-bit `full` mirror; `info` with the failed item unchanged and the other items bit-identical on
every family, both triangles, `full` or not, with and without `info`; the workspace per family (none for
cta and for blocked at n <= nb, below `n (nb + n)` per item above it); a forced three-item
chunk whose ten items are each verified; the cta pin launched at its advertised ceiling (graded input)
and refused one past it; 1024 identical items bit-identical to item 0 per family at orders whose CTA
work-groups span several sub-groups (n = 37, 100); a 70001-item batch; a short `info` span refused; and
the `reached` coverage rows of a child process: `cta` and `blocked:nb=64`, blocked's children exactly
trmm, gemm and herk (complex) or syrk (real), cta's none.

The potrs cases in the same file (`A X = B`, the bit-for-bit comparison with posv, `info` through posv)
name their own breaks above each test.

Deliberate breaks, planted one at a time in a second build tree and reverted (md5-checked), 2026-10-26,
the whole binary on CUDA and NETLIB, four types (green = passing of 112 cases; configs `fdcz` x `C`UDA /
`N`ETLIB). Cta rows two to seven and the last two ran against the suite of commit `de362488` (blocked
pinned at nb = 32 and 64); the rest against the final suite. Every cta break
turns blocked red too where blocked's diagonal blocks use the same kernel mode; every blocked break leaves
the cta-only cases green.

| break | red (test: configs) | green |
|---|---|---|
| `potri_cta.cc` trti2: the in-register substitution skips its last sub-diagonal term (`c < x` -> `c + 1 < x`) | Beyond: all types CUDA+NETLIB; Full: all types CUDA+NETLIB; Graded: all types CUDA+NETLIB; Inverts: all types CUDA+NETLIB; Parts: all types CUDA+NETLIB; Saturating: all types CUDA+NETLIB; Straddle: all types CUDA+NETLIB | 56 of 112 |
| `potri_cta.cc` trti2: the rank-TS update drops P(k, k) = 1 of the diagonal tile (`y <= c` -> `y < c`) | Full: all types CUDA+NETLIB; Graded: all types CUDA+NETLIB; Inverts: all types CUDA+NETLIB; Parts: all types CUDA+NETLIB; Saturating: all types CUDA+NETLIB; Straddle: all types CUDA+NETLIB | 64 of 112 |
| `potri_cta.cc` lauu2: the sum starts one row late (`k = ti` -> `k = ti + 1`) | Beyond: all types CUDA+NETLIB; Full: all types CUDA+NETLIB; Graded: all types CUDA+NETLIB; Inverts: all types CUDA+NETLIB; Parts: all types CUDA+NETLIB; Saturating: all types CUDA+NETLIB; Straddle: all types CUDA+NETLIB | 56 of 112 |
| `potri_cta.cc`: Upper loads `L = U^T` instead of `U^H` | Beyond: cz CUDA+NETLIB; Full: cz CUDA+NETLIB; Graded: cz CUDA+NETLIB; Inverts: cz CUDA+NETLIB; Parts: cz CUDA+NETLIB | 92 of 112 |
| `potri_cta.cc` store: the mirrored half without `conj` | Beyond: cz CUDA+NETLIB; Full: cz CUDA+NETLIB; Graded: cz CUDA+NETLIB; Inverts: cz CUDA+NETLIB; Parts: cz CUDA+NETLIB | 92 of 112 |
| `potri_cta.cc` trti2: the per-tile-row barrier removed | Full: all types CUDA+NETLIB; Graded: all types CUDA+NETLIB; Info: all types CUDA; Inverts: all types CUDA+NETLIB; Parts: cN dC dN fN zC zN; Saturating: all types CUDA+NETLIB; Straddle: cC cN dC dN fN zC zN | 63 of 112 |
| `potri_cta.cc`: info 0-based | Info: all types CUDA+NETLIB | 104 of 112 |
| `potri_blocked.hh` trtri Lower: `X21 = -W X11` (trmm, Right) replaced by a plain copy of W | Full: all types CUDA+NETLIB; Graded: all types CUDA+NETLIB; Inverts: all types CUDA+NETLIB; Parts: all types CUDA+NETLIB; Saturating: all types CUDA | 76 of 112 |
| `potri_blocked.hh` trtri Upper: the same for `X12 = -W X22` | Full: all types CUDA+NETLIB; Graded: all types CUDA+NETLIB; Inverts: all types CUDA+NETLIB; Parts: all types CUDA+NETLIB | 80 of 112 |
| `potri_blocked.hh` lauum Lower: the gemm leg dropped | Coverage: all types NETLIB; Full: dz CUDA+NETLIB; Graded: all types CUDA+NETLIB; Inverts: dz CUDA+NETLIB | 92 of 112 |
| `potri_blocked.hh` lauum: the copy of W back into A dropped | Full: all types CUDA+NETLIB; Graded: all types CUDA+NETLIB; Inverts: all types CUDA+NETLIB; Parts: all types CUDA+NETLIB; Saturating: all types CUDA | 76 of 112 |
| `potri_blocked.hh` lauum Lower: herk/syrk on the diagonal block dropped | Coverage: all types CUDA+NETLIB; Full: dz CUDA+NETLIB; Graded: all types CUDA+NETLIB; Inverts: dcz CUDA+NETLIB; Parts: dcz CUDA+NETLIB; Saturating: dcz CUDA | 77 of 112 |
| `potri_blocked.hh`: every batch part starts at item 0 | Parts: all types CUDA+NETLIB; Saturating: z CUDA | 103 of 112 |
| `potri_native.cc` `potri_diag_info` (removed since: the `Trtri` launch writes `info`): 0-based | Info: all types CUDA+NETLIB | 104 of 112 |
| `potri_blocked.hh` lauum: the last `Lauum` launch never restores the stash (run 2026-10-26, 88-case suite) | Info: all types CUDA+NETLIB | 80 of 88 |
| `potri_native.cc` `potri_finish`: flagged items not skipped (run 2026-10-26, 88-case suite) | Info: all types CUDA+NETLIB (the `full` cases) | 80 of 88 |
| `potri_native.cc` `potri_finish`: the mirror without `conj` | Full: cz CUDA+NETLIB; Parts: cz CUDA+NETLIB | 104 of 112 |

Two blind spots found on the way: the diagonally dominant inputs (condition below 8, off-diagonals
`1/n`) passed a broken intra-block substitution at the float cta ceiling on the host queue (n = 176) and
a dropped lauum gemm for float and cfloat; the graded inputs caught both, so the ceiling launch now runs
on a graded matrix.

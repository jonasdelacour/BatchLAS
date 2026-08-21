# WP4 Phase 2 -- benchmarking the blocked native POTRF against cuSOLVER

Answer first, because it decides whether Phase 3 routing is worth doing.

**The blocked native driver is NEVER faster than cuSOLVER on any cell whose
answer was right, and in a vendor-free build it is 2x slower on geometric
average.** The best reliably-correct cell in the whole campaign is 0.996x
(double, n=512, batch=128, three passes). Every ratio at or above 1.00 anywhere
in these files comes from a run that computed WRONG Cholesky factors. Geomean of
`cuSOLVER_ms / blocked_ms` over all 40 cells of the grid, >1 meaning the
native driver wins:

| configuration under the driver | float | double | cfloat | cdouble | all |
|---|---|---|---|---|---|
| `def` -- what `BATCHLAS_POTRF_ROUTE=blocked` gives today (vendor present) | 0.63 | 0.80 | 0.74 | 0.79 | **0.74** |
| `nn` -- both injected calls native: **the vendor-free build** | 0.61 | 0.83 | 0.41 | 0.36 | **0.52** |
| `VV` -- both injected calls vendor: the driver's own schedule cost | 0.56 | 0.69 | 0.40 | 0.46 | **0.51** |

(Restricted to cells whose answer was also CORRECT, the same geomeans are
`def` 0.58 over 11 cells, `nn` 0.52 over 13, `VV` 0.51 over all 40. That filter
biases toward small batch -- exactly where the defect below stops firing -- so
both are reported throughout and neither is used alone.)

And a defect that outranks all of it:

**The vendor-free blocked potrf RETURNS WRONG ANSWERS at large batch, on the
default route, in `build-novendor`, with no environment variable set.** It is the
native panel solve -- and specifically the V1 CTA kernel underneath
`trsm_native_blocked`, not that function's blocking, which sections 3 and 7
separate. It is non-deterministic, it fires at batch >= 64..256 depending on type
and order (i.e. exactly the regime this project cares about), no legal `nb`
avoids it, and the failing column is 32-aligned in 69% of 2070 recorded cases.
All four scalar types.

The five questions this was asked, and where each is answered:

| question | section |
|---|---|
| ratio per cell | 5, full table; `summary_main.txt` for every column |
| geomean | the table above; `summary_main.txt` for both filters |
| crossover | 5, "The crossover" -- it is in ORDER, not in batch |
| is it ever FASTER than cuSOLVER | 5, "Is native blocked potrf ever FASTER" -- no |
| where does the time go | 6, nsys per-stage split |
| nb sweep | 7 -- all four shipped nb confirmed; float's W is not |
| the route pin, verified | 2 |

---

## 1. What was measured, and why this way

`potrf` itself, never a synthetic gemm or trsm harness. Every operand the
blocked driver hands `gemm` and `trsm` is a SUB-VIEW of the parent carrying the
PARENT `ld` and the PARENT batch stride, and the native GEMM fast paths are gated
on `is_contiguous_dense_matrix` (`register_tiled_common.hh:74-77`), which such a
view fails by construction. WP3 measured the identical operand shapes at
0.86-0.98x of cuBLAS when `ld == rows` and 0.43-0.62x at the real `ld`. `OpShape`
carries no leading dimension (`route.hh:227-241`), so the router cannot see the
case at all. Only potrf issues those views; only potrf can measure the effect.

The PARENT here is the natural user shape, `ld == n`, `stride == n*n` -- padding
the parent would be measuring a different question. The strided-ld effect is
internal and unavoidable: at n=1024, nb=128 the driver hands gemm an 896-row
operand carrying `ld = 1024`, and hands trsm a 128x128 triangle carrying the same.
`ld != rows` for every sub-view of every panel, by construction, whatever the
caller's parent looks like.

Three arms, in ONE process, on ONE allocation, INTERLEAVED rep by rep:

* `vendor` -- `backend::potrf_vendor<CUDA,T>`, i.e. cuSOLVER.
* `blocked` -- `sycl_potrf::potrf_blocked_dispatch<T>` with the ROUTED `gemm` and
  the ROUTED `trsm` injected: byte for byte what the facade's
  `Algorithm::Blocked` arm does (`factorization.cc:261-276`).
* `cta` -- `sycl_potrf::potrf_cta_dispatch<T>`, whenever the order fits the
  Phase 1 tier, so "what does vendor freedom cost at this order" has one answer
  and not two.

A DIRECT call, not the facade, for the reason `potrf_native.hh` gives: a forced
route that `supports()` rejects falls back to `automatic()`
(`route_resolve.hh:101,:111) and silently runs cuSOLVER, so an env-pinned
benchmark can be timing the vendor while believing it is timing the kernel. A
direct call cannot be served by a vendor. (`bench facade` and `route.txt` check
the pin separately -- section 2.)

Which of the two injected calls lands on the vendor and which on the native
kernel is then set by `BATCHLAS_GEMM_ROUTE` / `BATCHLAS_TRSM_ROUTE`, giving three
configurations of the SAME driver:

| cfg | gemm | trsm | what it is |
|---|---|---|---|
| `def` | unset | unset | what `BATCHLAS_POTRF_ROUTE=blocked` delivers in a vendor-present build **today**. NOT all-vendor: the panel trsm already resolves `Native:Blocked` for double/complex at every batch and for float at batch >= 128 (`route_trsm.hh:304`), and the trailing gemm already resolves `Native:RegisterTiled` for double at batch >= 64 (`route_gemm.hh:122,:206`). |
| `nn` | native | native | **the vendor-free configuration** -- verified equal to `build-novendor`, section 4. |
| `VV` | vendor | vendor | correctness control, and the pure cost of the driver's schedule against cuSOLVER's. |

That `def` is not all-vendor is not read off the route table alone: 28 of the 40
`def` cells in `main.csv` return `info != 0` (and 27 of 40 `nn`, against 0 of 40 `VV`), and section 3 shows the native trsm
is the only thing in this driver that does that. The default route is already
reaching it.

### Hygiene

* GPU 1 of 2x RTX 4090 (128 SM), held for the whole campaign;
  `CUDA_VISIBLE_DEVICES=1`. GPU 0 was observed at 100% utilisation with 15 GB in
  use by a process outside this session during part of the campaign, which is
  why nothing was ever run there -- two agents benchmarking at once is only
  race-free because each owns a different physical card.
  `gpu_guard.sh` is in this directory and was used for the interactive probes.
  The batch scripts pin `CUDA_VISIBLE_DEVICES` directly rather than wrapping
  every one of the ~400 short processes in the guard's 5 s idle poll; GPU 1 was
  spot-checked with `nvidia-smi --query-compute-apps` at several points during
  the campaign and no foreign process was ever seen on it. That is weaker than
  the guard's post-run check and is stated as such.
* JIT warmed before every timed loop: `BENCH_WARM_S` (1.5-2.0 s) of
  restore+call cycles per arm, run AFTER a correctness pass, so clocks are
  already up when the first timed rep starts. An idle 4090 sits at 210 MHz.
* Interleaved A/B: rep r times arm A then arm B, never all of A then all of B.
* **Discard rule: any row with `rel_sd > 0.10` is dropped and NAMED** in
  `summary_main.txt`. Five of 244 rows in the main grid were dropped; all five
  were re-measured (`recheck.csv`) and one of them had been carrying a false
  claim -- see section 5.
* Never under `BATCHLAS_KERNEL_TRACE` (~60% inflation, and the implementer
  already recorded that it emits no kernel names in this build so it is useless
  as a route oracle anyway). Attribution is by `nsys` instead.
* The matrix is restored from a pristine device copy before EVERY timed call
  (`sq.memcpy(...).wait()` outside the timing window), so no call ever
  factorises an already-factorised matrix.
* Every arm's residual, upper-triangle preservation, non-finite count and `info`
  are checked in the same process on the same buffer BEFORE any timing. **No
  timed cell in these files can be a wrong answer without saying so.**

### The input

`A = n*(1 + 0.01*(b mod 17)) * I + R` with `R` Hermitian, entries uniform in
[-0.5, 0.5], the same `R` for every item. The off-diagonal spectral radius is
~0.6*sqrt(n) (26 at n=2048) against a diagonal of n, so the condition number is
under 1.05 and **no correctly implemented Cholesky can fail on it**. That is what
makes `info != 0` a driver defect rather than a property of the matrix -- and the
vendor arm, on the identical buffer in the identical process, is the standing
control for the claim: it returns `info == 0` and residual 1e-10 / 1e-19 in every
cell of every file here.

The whole allocation, padding included, is poisoned with junk first, so a driver
that reads outside its named window produces garbage rather than a plausible
answer. Sub-views in the harness are built with the explicit 6-argument
constructor carrying parent ld AND stride AND batch AND an own pointer array
(the `[FIX-B-trap]` workaround), never `operator()(Slice,Slice)`.

The failures in section 3 were FIRST found under a different input (`G = M^H M`
Gram fill, condition number 1.33) and reproduce identically under this one, so
nothing about them is an artefact of the fill.

## 2. The route pin, asked and not assumed

`route.txt`: the resolver's own answer for all four types x six orders x six
values of `BATCHLAS_POTRF_ROUTE`.

```
route  float  n=512  env=(unset)         resolved=Vendor:Auto     nb=128 W=32 cta_max_n=155
route  float  n=512  env=blocked         resolved=Native:Blocked
route  float  n=512  env=native:blocked  resolved=Native:Blocked
route  float  n=512  env=cta             resolved=Vendor:Auto        <-- the trap, live
route  float  n=512  env=vendor          resolved=Vendor:Auto
route  float  n=512  env=tyop            resolved=Vendor:Auto        <-- the trap, live
```

Two live demonstrations of the recorded trap in one table: a TYPO silently means
vendor, and so does a legitimate-looking `cta` at any order above
`potrf_cta_max_n<T>()` (`supports()` rejects it and `automatic()` returns the
vendor). `blocked` and `native:blocked` do take effect at every order and type
tested, 64 through 2048.

The shipped block widths, as the driver itself reports them
(`potrf_blocked_debug_params`): nb/W = float 128/32, double 96/32, cfloat 96/32,
cdouble 64/16, and nb collapses to `min(nb, n)` at small n.

### And the pin taken through the real facade

`route.txt` says the resolver returns `Native:Blocked`; that is not the same as
the facade running it. `batchdep.csv` was collected in `bench facade` mode --
the public `potrf<CUDA,T>` with `BATCHLAS_POTRF_ROUTE=blocked` -- and its medians
match the direct-call `def` medians of `main.csv` on every shared cell:

| type | n | batch | facade | direct call | ratio |
|---|---|---|---|---|---|
| float | 512 | 128 | 1.543 | 1.546 | 0.998 |
| float | 1024 | 128 | 7.712 | 8.065 | 0.956 |
| float | 1024 | 256 | 15.787 | 15.955 | 0.990 |
| cdouble | 512 | 128 | 25.713 | 25.864 | 0.994 |
| cdouble | 1024 | 128 | 181.249 | 182.244 | 0.995 |
| cdouble | 1024 | 256 | 347.382 | 348.919 | 0.996 |

Both directions are covered by that: the facade is not silently running
cuSOLVER (it is 2-24x off cuSOLVER's time in these cells), and the direct call
is not measuring something the facade cannot reach. The single 4.4% outlier is
float 1024/128, where both figures come from a single 5-rep and a single 2-rep
process respectively and neither was re-measured; it is not a cell any claim in
this file rests on.

## 3. THE BLOCKING DEFECT: the vendor-free potrf is WRONG at large batch

Found while benchmarking, not looked for.

`novendor.csv` -- **`build-novendor`, no environment variable set at all, the
genuine vendor-free default path** -- reports `info != 0` on every type:

| type | n | batch | items failing of batch | residual of first failing item |
|---|---|---|---|---|
| float | 1024 | 128 | 6 | 5.3e-02 |
| float | 1024 | 256 | 63 | 1.3e+01 |
| float | 2048 | 64 | 15 | 1.0e-04 |
| double | 512 | 128 | 1 | 4.4e+03 |
| double | 1024 | 256 | 18 | 5.5e-01 |
| double | 2048 | 64 | 7 | 4.9e-04 |
| cfloat | 1024 | 128 | 10 | 2.8e-02 |
| cfloat | 1024 | 256 | 30 | 9.0e-02 |
| cfloat | 2048 | 64 | 16 | 4.9e-04 |
| cdouble | 256 | 256 | 4 | 4.5e-02 |
| cdouble | 512 | 128 | 2 | 5.2e-02 |
| cdouble | 1024 | 128 | 25 | 5.0e-02 |

On an input whose condition number is under 1.05 and on which cuSOLVER returns
`info == 0` and residual 1e-10 in the same process.

### It is the panel TRSM, not the trailing GEMM and not the driver

`allvendor.csv` forces each injected call independently. Three repeats per cell:

| type | n | batch | `VV` gemm=V trsm=V | `nV` gemm=N trsm=V | `Vn` gemm=V trsm=N | `nn` gemm=N trsm=N |
|---|---|---|---|---|---|---|
| float | 1024 | 128 | 0,0,0 | 0,0,0 | 2,1,2 | 9,10,11 |
| float | 1024 | 256 | 0,0,0 | 0,0,0 | 28,32,31 | 60,65,50 |
| float | 512 | 256 | 0,0,0 | 0,0,0 | 1,2,0 | 0,0,0 |
| cdouble | 1024 | 128 | 0,0,0 | 0,0,0 | 36,33,36 | 28,26,26 |
| cdouble | 1024 | 256 | 0,0,0 | 0,0,0 | 74,89,81 | 60,64,64 |
| cdouble | 512 | 256 | 0,0,0 | 0,0,0 | 45,48,49 | 40,39,23 |

(cells are `info_nonzero` out of `batch`, three repeats)

**Every cell with the vendor trsm is clean. Every cell with the native trsm
fails.** The native GEMM is innocent -- `nV` is clean everywhere, including the
cells where `nn` fails hardest. So the driver's schedule, its fixup/quench
kernel, its info merge and its fold are all exonerated by the same table.

### It is batch-dependent and non-deterministic

`batchdep.csv`, through the PUBLIC FACADE with `BATCHLAS_POTRF_ROUTE=blocked`,
three repeats:

| type | n | batch=1 | 8 | 32 | 64 | 96 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| float | 512 | 0,0,0 | 0,0,0 | 0,0,0 | 0,0,0 | 0,0,0 | 0,0,0 | 1,1,3 |
| float | 1024 | 0,0,0 | 0,0,0 | 0,0,0 | 0,0,0 | 0,0,0 | 0,1,1 | 30,24,25 |
| cdouble | 512 | 0,0,0 | 0,0,0 | 0,0,0 | 0,0,0 | 0,2,0 | 6,3,5 | 46,47,52 |
| cdouble | 1024 | 0,0,0 | 0,0,0 | 0,0,0 | 6,4,4 | 22,25,28 | 33,21,37 | 85,86,85 |

Three properties worth stating separately.

1. **A small-batch verification cannot see it.** Everything at batch <= 32 is
   clean for both types at both orders. The implementer's proof run was at batch
   128 and n <= 1000, which lands on the very edge: float n=1024 batch=128 fails
   0 or 1 items out of 128 depending on the run.
2. **It is non-deterministic.** Four runs of one identical command gave 8, 12, 15
   and 17 failing items. A repeat that comes back clean therefore proves nothing.
3. **The failing column is 32-ALIGNED**, and that is the sharpest pointer at the
   mechanism this benchmark can give. Over all 2070 `info` values recorded in
   the `*.err` files of this directory, `(info - 1) mod 32`:

   | residue | 0 | 1 | 2 | 3 | rest |
   |---|---|---|---|---|---|
   | count | 1436 | 307 | 105 | 67 | 155 |

   69% land exactly on a multiple of 32 and 93% within 3 of one, against 12.5%
   for the uniform null. 32 is `trsm_cta_max_n<T>()` for every type -- the V1
   block width `trsm_native_blocked` cuts its triangle into. So the leaf is
   finding its first non-positive pivot at a column where a V1 BLOCK BOUNDARY of
   the preceding panel solve falls, not at a random column, and the corruption
   is per-32-column-block of the panel rather than smeared over it. `(info-1) mod
   nb` puts the same mass on 0, 32 and 64 (1305 / 67 / 60), i.e. block-interior
   32-boundaries as well as block starts.

### Relation to what the measure phase already recorded

The measure phase found this kernel wrong and recorded a mitigation: keep `nb` a
multiple of `trsm_cta_max_n<T>()` = 32, because orders 48, 77, 80 and 109 fail
90-128 of 128 items DETERMINISTICALLY while 32, 64, 96 and 128 are clean. The
driver implements exactly that (`potrf_blocked.cc`, the round-down in
`potrf_blocked_params`), and every nb here -- 128, 96, 64 -- is a legal multiple.

**The mitigation removed the deterministic mode and not the sporadic one.** The
same measure-phase note records that orders 128 and 160 "fail 0-5 items
sporadically, non-deterministic across repeats", and that sporadic mode is what
this file measures: it does not go away with a legal nb, and it grows with batch
until it is 85 of 256 items. Reading it as "contained" was the error; it is
contained only below batch 64.

### Consequence for the numbers in this file

A cell whose blocked arm failed is marked `WRONG` in `summary_main.txt` and
contributes to no correctness-filtered geomean. Its TIMING is still reported and
still used in the unfiltered geomean, and that is deliberate: a right-looking
Cholesky has no pivoting and no early exit, so a wrong run issues the same
kernels on the same shapes and its wall time is a valid timing of the same work.
Filtering to correct cells alone would bias every summary toward SMALL BATCH,
which is exactly where the defect stops firing -- and, separately, where this
driver looks worst. Both geomeans are printed for that reason.

One caveat that cannot be resolved from here, and it is the pivotal one for
Phase 3: **the native trsm is both faster and wrong.** nsys puts the panel solve
at 4.05 ms/call native against 6.11 vendor for float and 16.5 against 144.7 for
cdouble, at n=1024 batch=256 (section 6). Every cell in these files that reaches
cuSOLVER parity does so on that kernel (section 5). Whether a corrected V1 solve
keeps the advantage is unknown, so **the `nn` and `def` numbers below are
optimistic by an unknown amount**, and the `VV` column -- the same driver with a
correct panel solve -- is the pessimistic bound.

## 4. Forced `native` IS what `build-novendor` runs -- verified, not assumed

This repository has shipped four bugs from the forced/supported distinction, so
the `nn` configuration was not taken on trust. `novendor.csv` re-runs the grid in
`build-novendor` (`-DBATCHLAS_ENABLE_VENDOR_BLAS=OFF`) with NO env at all, which
reaches native through the vendor-free fallback (`route_resolve.hh:60-63`) rather
than through forcing. Median ms, `nn` (vendor-present, forced) against
`novendor` (vendor-free, unforced):

| type | n | batch | `nn` | `novendor` | ratio |
|---|---|---|---|---|---|
| float | 256 | 256 | 0.7032 | 0.7037 | 1.001 |
| float | 512 | 128 | 1.7109 | 1.7177 | 1.004 |
| float | 1024 | 128 | 9.1070 | 9.1320 | 1.003 |
| float | 1024 | 256 | 17.2400 | 17.2743 | 1.002 |
| float | 2048 | 64 | 29.9591 | 30.0615 | 1.003 |
| double | 256 | 256 | 2.1230 | 2.1233 | 1.000 |
| double | 512 | 128 | 6.0070 | 6.0019 | 0.999 |
| double | 1024 | 128 | 39.7714 | 40.0579 | 1.007 |
| double | 1024 | 256 | 78.9873 | 78.9288 | 0.999 |
| double | 2048 | 64 | 152.5429 | 153.0467 | 1.003 |
| cfloat | 128 | 512 | 0.6088 | 0.6067 | 0.997 |
| cfloat | 256 | 256 | 1.3012 | 1.2867 | 0.989 |
| cfloat | 512 | 128 | 4.0920 | 4.1133 | 1.005 |
| cfloat | 1024 | 128 | 28.1947 | 28.2310 | 1.001 |
| cfloat | 1024 | 256 | 55.6394 | 56.0006 | 1.006 |
| cfloat | 2048 | 64 | 111.5201 | 111.3279 | 0.998 |
| cdouble | 128 | 512 | 4.4401 | 4.4473 | 1.002 |

Worst disagreement 1.1%, across a different binary and a different resolution
path. `nn` is a faithful stand-in for the vendor-free build, and the vendor arm
interleaved beside it in the same process is therefore a fair reference.

(Two `novendor` rows were discarded by the sd rule: `double 128/512`
rel_sd = 1.15 and `float 2048/64` rel_sd = 0.119. They are excluded from the
table above.)

## 5. The main grid

Full per-cell table with every column in `summary_main.txt`; raw rows in
`main.csv`. `_x` is `vendor_ms / blocked_ms`, so **>1 means the native driver is
faster than cuSOLVER**.

A `+` marks a cell replaced by `overrides.csv` -- the median of three
independent passes of 7 or 9 reps, from `recheck.csv` (cells the sd rule
discarded, and two that disagreed with themselves) or `wins.csv` (every cell that
read >= 1.00). `main.csv` is left exactly as the run produced it so the
difference stays auditable; `make_overrides.py` builds the override file and
each row in it names its source and pass count. The geomeans below use the
overridden values.

```
type         n  batch   nb   W  vendor_ms     def_ms   def_x      nn_ms    nn_x      VV_ms    VV_x    cta_ms   cta_x  flags   (+ = value from overrides.csv)
cdouble    128    512   64  16     1.9608    3.1215    0.628    4.4401    0.439    8.5525    0.228        --      --  def=WRONG(info=6,res=3.4e-18)
cdouble    128   2048   64  16     7.6545   12.5160    0.612   17.7759    0.430   14.4618    0.529        --      --  def=WRONG(info=49,res=3.5e-01); nn=WRONG(info=33,res=1.2e-02)
cdouble    256    256   64  16     6.4971    7.3589    0.883   15.1244    0.429   25.5089    0.255        --      --  def=WRONG(info=5,res=8.7e-19); nn=WRONG(info=4,res=8.7e-19)
cdouble    256   1024   64  16    25.7749   31.1784    0.827   62.0444    0.415   49.0082    0.526        --      --  def=WRONG(info=80,res=2.1e-02); nn=WRONG(info=67,res=4.3e-02)
cdouble    512    128   64  16    22.1171   25.8645    0.855   60.4904    0.365   68.1886    0.325        --      --  def=WRONG(info=3,res=4.3e-19); nn=WRONG(info=3,res=4.3e-19)
cdouble    512    512   64  16    87.5316   98.0245    0.893  240.2077    0.364  152.0043    0.587        --      --  def=WRONG(info=91,res=4.7e+01); nn=WRONG(info=91,res=9.5e-02)
cdouble   1024    128   64  16   155.2767  182.2435    0.852  481.9013    0.328  272.9749    0.582        --      --  def=WRONG(info=37,res=3.4e+05); nn=WRONG(info=34,res=1.5e+00)
cdouble   1024    256   64  16   310.0565  348.9194    0.889  955.1269    0.324  462.4477    0.670        --      --  def=WRONG(info=57,res=2.1e-04); nn=WRONG(info=43,res=4.3e-19)
cdouble   2048     32   64  16   300.3301  423.1463    0.710 1070.6452    0.280  609.1380    0.493        --      --  def=WRONG(info=5,res=3.8e-19); nn=WRONG(info=9,res=4.9e-04)
cdouble   2048     64   64  16   581.8610  733.0554    0.794 1988.8318    0.293  903.3367    0.644        --      --  def=WRONG(info=35,res=2.2e-01); nn=WRONG(info=32,res=1.8e-03)
cfloat     128    512   96  32     0.1918    0.5124    0.374    0.6088    0.314    1.1110    0.173        --      --
cfloat     128   2048   96  32     0.9484    2.1948    0.432    2.4650    0.384    2.5937    0.364        --      --  def=WRONG(info=68,res=1.4e+01); nn=WRONG(info=44,res=1.7e-09)
cfloat     256    256   96  32     0.4975    0.8166    0.609    1.3012    0.378    2.0816    0.237        --      --  def=WRONG(info=4,res=7.0e-10)
cfloat     256   1024   96  32     2.6483    3.5293    0.750    5.0838    0.526    7.5936    0.349        --      --  def=WRONG(info=93,res=2.0e-02); nn=WRONG(info=84,res=3.6e+01)
cfloat     512    128   96  32     1.5798    2.0741    0.762    4.0920    0.386    5.1887    0.303        --      --  def=WRONG(info=3,res=4.3e-10)
cfloat     512    512   96  32     8.0023    8.4023    0.952   15.3665    0.524   19.1952    0.416        --      --  def=WRONG(info=93,res=3.8e-01); nn=WRONG(info=89,res=6.9e-10)
cfloat    1024    128   96  32    11.9991   11.8127+   1.023   28.2070+   0.427   18.5553+   0.649        --      --  def=WRONG(info=38,res=3.2e-10); nn=WRONG(info=18,res=3.2e-10)
cfloat    1024    256   96  32    27.1630   25.6343+   1.075   55.5863+   0.493   48.6759+   0.562        --      --  def=WRONG(info=73,res=4.5e+00); nn=WRONG(info=33,res=2.5e-01)
cfloat    2048     32   96  32    19.6996   23.9440    0.823   60.3494    0.328   36.8721    0.536        --      --  def=WRONG(info=2,res=1.0e-10); nn=WRONG(info=3,res=1.0e-10)
cfloat    2048     64   96  32    43.7464   44.3461    0.986  111.5201    0.392   57.8048    0.753        --      --  def=WRONG(info=33,res=1.4e+08); nn=WRONG(info=14,res=1.0e-10)
double     128    512   96  32     0.5904    1.1791    0.501    1.1744    0.498    1.0496    0.559        --      --
double     128   2048   96  32     2.1518    4.7663    0.451    4.7658    0.451    4.1766    0.515        --      --  def=WRONG(info=59,res=6.8e+05); nn=WRONG(info=46,res=3.2e-18)
double     256    256   96  32     1.8075    2.1233+   0.851    2.1229+   0.851    2.6697+   0.677        --      --
double     256   1024   96  32     6.8854    8.3296    0.827    8.3403    0.827   10.5448    0.650        --      --  def=WRONG(info=86,res=4.9e+00); nn=WRONG(info=72,res=1.9e+01)
double     512    128   96  32     6.4353    6.0051+   0.996    6.0081+   0.996    7.9675+   0.750        --      --
double     512    512   96  32    22.8321   23.1328+   0.980   23.1263+   0.980   30.7874+   0.735        --      --  def=WRONG(info=81,res=4.3e-19); nn=WRONG(info=83,res=4.3e-19)
double    1024    128   96  32    40.3179   40.0192+   1.007   40.0603+   1.006   50.7191+   0.793        --      --  def=WRONG(info=4,res=4.0e-19); nn=WRONG(info=4,res=4.0e-19)
double    1024    256   96  32    79.8070   78.7849+   1.012   78.7564+   1.013   99.8934+   0.798        --      --  def=WRONG(info=20,res=2.2e-19); nn=WRONG(info=24,res=2.2e-19)
double    2048     32   96  32    78.0661  107.6462+   0.727   80.3007+   0.976  114.2439+   0.685        --      --
double    2048     64   96  32   149.7298  152.9819    0.979  152.5429    0.982  194.6417    0.768        --      --  def=WRONG(info=3,res=1.1e-19); nn=WRONG(info=7,res=1.1e-19)
float      128    512  128  32     0.1397    0.3017+   0.463    0.3019+   0.466    0.3019+   0.465    0.2927   0.477
float      128   2048  128  32     0.4642    1.4588    0.318    1.4630    0.319    1.4636    0.325    1.4503   0.320
float      256    256  128  32     0.3366    0.6282    0.536    0.7032    0.477    0.7142    0.469        --      --
float      256   1024  128  32     1.4221    2.5319    0.562    2.8238    0.508    3.1273    0.456        --      --  def=WRONG(info=39,res=9.0e-10); nn=WRONG(info=28,res=9.0e-10)
float      512    128  128  32     0.9018    1.5462    0.583    1.7109    0.524    1.8590    0.485        --      --
float      512    512  128  32     4.7035    5.6235    0.836    6.3669    0.741    7.4116    0.650        --      --  def=WRONG(info=47,res=4.6e-10); nn=WRONG(info=53,res=4.6e-10)
float     1024    128  128  32     7.3358    8.0651    0.910    9.1070    0.809    9.6664    0.765        --      --  def=WRONG(info=1,res=1.1e-10); nn=WRONG(info=14,res=1.1e-10)
float     1024    256  128  32    17.3315   15.9369+   1.118   17.2566+   1.017   19.8097+   0.887        --      --  def=WRONG(info=31,res=5.8e-11); nn=WRONG(info=69,res=5.8e-11)
float     2048     32  128  32    13.2529   22.5192    0.589   18.8512    0.702   22.4805    0.586        --      --
float     2048     64  128  32    27.4761   34.0234    0.808   29.9591    0.913   34.0836    0.810        --      --  nn=WRONG(info=13,res=5.8e-11)
```

### Is native blocked potrf ever FASTER than cuSOLVER?

**No -- not once, in any configuration, on any cell whose answer was right.**

Every cell in `main.csv` that read >= 1.00 was re-measured with 9 reps in 3
independent passes (`wins.csv`), and one that had been discarded by the sd rule
was re-measured too (`recheck.csv`). Median of the three passes:

| type | n | batch | `def` | `nn` | `VV` (correct) | answer in `def`/`nn` |
|---|---|---|---|---|---|---|
| float | 1024 | 256 | **1.118** | **1.017** | 0.887 | 25-31 / 69 items wrong, every pass |
| double | 1024 | 256 | **1.012** | **1.013** | 0.798 | 16-20 / 10-24 wrong, every pass |
| double | 1024 | 128 | **1.007** | **1.006** | 0.794 | 0-4 / 1-4 wrong |
| double | 512 | 512 | 0.980 | 0.980 | 0.735 | 69-81 / 67-83 wrong |
| cfloat | 1024 | 256 | **1.075** | 0.493 | 0.562 | 60-73 wrong, every pass |
| cfloat | 1024 | 128 | **1.023** | 0.427 | 0.649 | 34-38 wrong, every pass |
| double | 512 | 128 | 0.996 | 0.996 | 0.750 | **all clean, 3 passes** |
| double | 2048 | 32 | 0.727 | 0.976 | 0.685 | **all clean, 3 passes** |

**Every ratio at or above 1.00 comes from a run that computed wrong Cholesky
factors, and every one of them uses the native panel trsm.** Their correct
counterpart, the same driver with a correct panel solve, is 0.56-0.89x in the
same process. The apparent parity at n=1024 is bought with the defect.

The one apparent exception proves the rule rather than breaking it: `double
n=1024 batch=128 def` returned `info == 0` on pass 3 at 1.007x -- and returned 4
and 3 wrong items on passes 1 and 2 of the identical command. A clean run of a
non-deterministic defect is luck, not a configuration.

**Best reliably-correct cell in the whole campaign: 0.996x**, double n=512
batch=128, three passes, both `def` and `nn`. Second: 0.976x, double n=2048
batch=32, `nn`, three passes.

One row in the raw grid also had to be withdrawn, and it is an argument for the
discard rule rather than against it: `double n=512 batch=128 nn_x = 1.055` in
`main.csv` was FALSE. Its VENDOR arm had been discarded for rel_sd = 0.147, so
the ratio was computed against a contaminated reference. Re-measured over three
passes it is 0.996x. Without the recheck, a benchmark that had already flagged
and dropped the offending row would still have printed 1.055x as a win.

### The crossover

There is one, and it is in ORDER, not in batch: the ratio climbs with `n` to a
maximum around n=1024 and falls back at n=2048. Range over the two batches
measured at each order (rechecked values where a recheck exists):

| cfg | type | n=128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| `def` | float | 0.32-0.46 | 0.54-0.56 | 0.58-0.84 | 0.91-1.12 | 0.59-0.81 |
| `def` | double | 0.45-0.50 | 0.83-0.85 | 0.98-1.00 | 1.01 | 0.73-0.98 |
| `def` | cfloat | 0.37-0.43 | 0.61-0.75 | 0.76-0.95 | 1.02-1.08 | 0.82-0.99 |
| `def` | cdouble | 0.61-0.63 | 0.83-0.88 | 0.86-0.89 | 0.85-0.89 | 0.71-0.79 |
| `nn` | float | 0.32-0.47 | 0.48-0.51 | 0.52-0.74 | 0.81-1.02 | 0.70-0.91 |
| `nn` | double | 0.45-0.50 | 0.83-0.85 | 0.98-1.00 | 1.01 | 0.98 |
| `nn` | cfloat | 0.31-0.38 | 0.38-0.53 | 0.39-0.52 | 0.43-0.49 | 0.33-0.39 |
| `nn` | cdouble | 0.43-0.44 | 0.42-0.43 | 0.36 | 0.32-0.33 | 0.28-0.29 |

Two shapes in that table, not one.

* **The real types converge on cuSOLVER as the order grows.** double is within
  2% from n=512 up and float reaches 0.81-1.02 at n=1024. Whatever the driver
  costs, it is amortised.
* **Complex diverges.** cfloat and cdouble get monotonically WORSE with `n` in
  the vendor-free configuration -- cdouble halves from 0.44 to 0.28 between
  n=128 and n=2048. A driver that loses more as the problem grows is not paying
  a fixed overhead; it is running a kernel that does not scale with the problem,
  and section 6 names it: the Tiled16 complex GEMM fallback.

The `def` row for complex hides that, because `def` uses cuBLAS's gemm for
complex and only the trsm is native. Vendor-free, there is no such hiding.

### Below the CTA ceiling, for scale

n=128 fits the Phase 1 CTA tier for float. Both native tiers lose the same
amount: CTA 0.293 ms and blocked 0.301 ms against cuSOLVER's 0.140 ms at batch
512 (0.48x / 0.47x), and 1.450 / 1.459 against 0.464 at batch 2048 (0.32x). The
blocked driver at n <= nb is just the leaf plus one fixup launch, and the 2-3%
difference is that launch. **So the 2-3x deficit at small n is Phase 1's, not
Phase 2's**, and no amount of blocked-driver tuning addresses it.

## 6. Where the time goes

`nsys`, not `BATCHLAS_KERNEL_TRACE`: the implementer already recorded that the
trace emits only `sycl_submit` / `sycl_parallel_for` with no kernel names in this
build, so it cannot attribute time to a stage, and it inflates wall time ~60%
besides. Raw per-kernel rows in `nsys/*_cuda_gpu_kern_sum.csv`, bucketed by
`nsys_summary.py` into `nsys/split.txt`. Both arms are in the same profile and
are separated by kernel name; per-call figures come from dividing by a kernel
that runs exactly once per call.

Two caveats stated because they move the numbers by 20%:

* cuSOLVER's kernels include `potrf_cta_lower_batch` and `potrfBatch_trsm_lower`,
  which are one underscore away from BatchLAS's own `PotrfCtaKernel`. Confusing
  them puts 120 ms of vendor time inside the native leaf.
* the native panel trsm has its OWN injected trailing gemm (WP3), so in `nn` part
  of the `gemm` bucket belongs to the panel solve. They separate by LAUNCH COUNT:
  float n=1024 / nb=128 / W=32 issues exactly 217 trailing gemms per potrf call
  (112 W-wide diagonal blocks + 105 below-diagonal rectangles), and the profile
  shows 84 + 133 = 217, leaving the third entry (21) as the trsm's. The figures
  below are the SEPARATED ones.

### float, n=1024, batch=256 (ms per call)

| stage | `VV` | `nn` |
|---|---|---|
| trailing gemm | 10.95 (cuBLAS `ampere_sgemm_128x128_nt`, 217 launches) | **10.63** (`GemmRegisterTiled<float,128,32>` 84 launches + `GemmTiled16` 133) |
| panel trsm | 6.11 (cuBLAS `batch_trsm_right_kernel`) | **4.05** (`TrsmCtaKernel<float,32,Right>` 0.84 + its own injected gemm 3.21) |
| leaf (Phase 1 CTA) | 1.41 | 1.38 |
| fold | 0.64 (112 launches) | 0.61 |
| fixup / quench | 0.01 | 0.43 |
| **blocked total** | **19.17** | **17.10** |
| cuSOLVER, same process | 17.62 | 17.53 |

**The native float trailing GEMM is at PARITY with cuBLAS on the shapes the
driver actually issues** -- 10.63 against 10.95, a 3% win. That contradicts the
0.13-0.18x the measure phase recorded for float, and the reason is that the two
measured different shapes: the measure phase timed ONE trailing gemm of roughly
`m2 x m2 x nb`, while the driver's triangular decomposition issues 217 gemms of
`32 x 32 x 128` and `mr x 32 x 128`. The W-panelled shape mix is where the native
kernel is competitive. **A figure for "the trailing update" taken from a single
square gemm does not transfer to this driver**, and the injection seam
(`PotrfTrailingGemm`) is what makes that harmless: the router picks per call.

### cdouble, n=1024, batch=256 (ms per call)

| stage | `VV` | `nV` | `nn` |
|---|---|---|---|
| trailing gemm | 310.0 (`cutlass_80_tensorop_z884gemm`, 945 launches) | 886 | **~914** (`GemmTiledGeneral<complex<double>,16>`) |
| panel trsm | 144.7 | 144.7 | **~31** |
| leaf | 4.17 | 4.16 | 4.16 |
| fold | 1.88 | 1.83 | 1.84 |
| fixup | 0.02 | 0.02 | 0.50 |
| **blocked total** | **460.8** | **1036.3** | **951.8** |
| cuSOLVER, same process | 308.8 | 308.7 | 308.9 |

**The native complex GEMM is 2.95x slower than cuBLAS on these shapes and is
97.6% of the vendor-free cdouble factorisation.** It is
`GemmTiledGeneralKernel<std::complex<double>, 16>` -- the Tiled16 FALLBACK, which
is what complex always gets: `route_gemm.hh:113-114` returns `false` for complex
in `preferred()`, and the second gate, `select_kernel_variant`
(`gemm_kernels.cc:471`), has its whole register ladder inside
`if constexpr (is_same_v<T,float>)`, so complex falls to
`max_dim <= 64 ? Direct : Tiled16`. There is no register-tiled complex kernel to
route to.

Arithmetic on the consequence: substituting cuBLAS's gemm time into the otherwise
unchanged `nn` breakdown gives `951.8 - 914 + 310 = 348` ms, i.e. **0.89x** of
cuSOLVER instead of 0.32x. A register-tiled complex GEMM is therefore worth 2.7x
on vendor-free cdouble potrf on its own, and is the single highest-value item
this benchmark identifies.

For scale on how far off that is: the trailing update is
`4 * n^3/3 * batch = 366` GFLOP, so cuBLAS runs it at 1.18 TFLOP/s and cuSOLVER
does the entire factorisation at 1.31 TFLOP/s -- both essentially at this card's
~1.29 TFLOP/s FP64 ceiling. Tiled16 runs it at 0.40 TFLOP/s.

Note also what `VV`'s "vendor trsm" actually is for complex: the profile names
`batchlas::backend::trsm_vendor<CUDA, std::complex<double>>`, a BatchLAS SYCL
kernel, not a cuBLAS one. `src/backends/cublas.cc:1111-1218` intercepts complex before any
cuBLAS call and runs a hand-written serial substitution. It costs 144.7 ms/call,
31% of `VV`. **There is no vendor complex trsm on this backend**, so the "vendor"
column for complex is not what it looks like, and the 0.40 / 0.46 `VV` geomeans
for cfloat / cdouble are partly that fallback's fault rather than the driver's.

### The obvious hypothesis, measured and REJECTED

The driver issues 200-1000 kernel launches per call (float n=1024: 8 leaves,
8 fixups, 7 trsms, 217 gemms, 112 folds; cdouble n=1024: 960 gemms, 480 folds).
Launch overhead is the natural suspect. It is not the problem:

| cell | blocked GPU time/call (nsys) | blocked wall time/call (main.csv) | GPU busy |
|---|---|---|---|
| float 1024/256 `nn` | 17.10 | 17.24 | 99.2% |
| float 1024/256 `VV` | 19.17 | 19.83 | 96.7% |
| cdouble 1024/256 `nn` | 951.8 | 955.1 | 99.7% |
| cdouble 1024/256 `VV` | 460.8 | 462.4 | 99.7% |

The GPU is busy for 97-99.7% of the wall clock. Fusing launches, folding the
fold into the gemm epilogue, or cutting the panel count buys at most 3%. **The
deficit is in the KERNELS, not in the schedule around them** -- which is the
opposite of what a 1440-launch factorisation looks like it should be, and is why
it was measured rather than assumed.

## 7. The nb and W sweep

`nbsweep.csv`, on potrf itself through `BATCHLAS_POTRF_NB` / `BATCHLAS_POTRF_W`
(read by `potrf_blocked_params`), n=1024, batch=256, 4 reps.

**Read the requested nb against the used one.** `potrf_blocked_params` clamps
`nb` to `min(request, device ceiling, n)` and then rounds DOWN to a multiple of
`trsm_cta_max_n<T>()` = 32. The ceilings are {155, 109, 109, 77}, so a request of
128 or 160 becomes 128 for float, 96 for double and cfloat, 64 for cdouble. Rows
that look like distinct data points are the same configuration; only the
distinct values below are real. Non-multiples of 32 were deliberately NOT swept:
they silently round to a neighbour and would print a fake data point.

### nb, at the shipped W (median ms; `X` = the answer was wrong)

| cfg | type | nb=32 | 64 | 96 | 128 | shipped | verdict |
|---|---|---|---|---|---|---|---|
| `nn` | float | 40.86 | 30.19 X | 27.77 X | **17.27 X** | 128 | **confirmed**, and by 1.61x |
| `nn` | double | 85.93 X | 80.03 X | **78.96 X** | (=96) | 96 | **confirmed** |
| `nn` | cfloat | 66.07 | 57.77 X | **55.51 X** | (=96) | 96 | **confirmed** |
| `nn` | cdouble | 991.47 X | **955.06 X** | (=64) | (=64) | 64 | **confirmed** |
| `VV` | float | 29.80 | 22.17 | 23.06 | **20.43** | 128 | confirmed |
| `VV` | double | 108.87 | 101.63 | **100.26** | (=96) | 96 | confirmed |
| `VV` | cfloat | 40.78 | **35.38** | 49.06 | (=96) | 96 | **64 is 28% better with a vendor gemm** |
| `VV` | cdouble | **435.82** | 462.41 | (=64) | (=64) | 64 | **32 is 6% better with a vendor gemm** |

All four shipped `nb` values are confirmed optimal in the vendor-free
configuration, on the real driver, at the batch this project cares about. That
is a null result and it is worth having: the constants were chosen on a STAGED
driver at batch 128 and n <= 1024, and they survive the move to the real one.

The float mechanism the implementer named also survives: nb=128 is 1.61x better
than nb=96 and nothing else in the table has a step that size. `k == nb`, and
float's transposed register kernel needs `k >= 128`.

### W, at the shipped nb (median ms)

| cfg | type | W=16 | 32 | 64 | 128 | 256 | shipped | best |
|---|---|---|---|---|---|---|---|---|
| `nn` | float | 27.82 | 17.27 | 18.00 | **16.21** | 19.55 | 32 | **128, by 6.5%** |
| `nn` | double | **77.94** | 78.96 | 82.16 | 89.39 | 103.79 | 32 | 16, by 1.3% |
| `nn` | cfloat | 56.38 | **55.51** | 58.53 | 64.74 | 77.31 | 32 | 32 |
| `nn` | cdouble | **955.06** | 974.19 | 1018.32 | 1105.79 | 1269.97 | 16 | 16 |
| `VV` | float | 30.93 | 20.43 | 15.86 | **15.12** | 17.50 | 32 | **128, by 35%** |
| `VV` | double | 178.66 | **100.26** | 104.61 | 112.63 | 127.59 | 32 | 32 |
| `VV` | cfloat | 59.57 | 49.06 | **46.44** | 48.06 | 53.03 | 32 | 64 |
| `VV` | cdouble | 462.41 | **447.23** | 462.54 | 495.78 | 551.78 | 16 | 32 |

Three of the four shipped `W` values are confirmed. **float is not**: W=128 beats
W=32 by 6.5% vendor-free and by 35% with a vendor gemm, and the curve is
non-monotonic (32 good, 64 worse, 128 best), which is a kernel-variant boundary
and not the wasted-work fraction the constant was reasoned from. The
implementer's recorded caveat that "with the VENDOR gemm injected the optimum
moves up to 96-128" is CONFIRMED for float and cfloat and REFUTED for double and
cdouble, both of which want 32 either way.

Changing W is a driver constant and out of scope here (this is a benchmark, not
a tuning commit), but float W=128 is the one shipped number this grid says is
wrong.

### nb=32 does NOT dodge the trsm defect -- which localises it

At nb=32 the panel solve's triangular order is exactly `trsm_cta_max_n<T>()`, so
`trsm_native_blocked` does not block at all. Reading the loop at
`trsm_native.cc:780-800`: `outer_nb >= nb` gives `LO=0, HI=32`, the inner loop
gives `lo=0, hi=32`, `LO > 0` is false so there is no outer `apply_update`,
`lo > LO` is false so there is no inner one, and the body is a single
`solve_diag` -> `trsm_native_v1_dispatch`. **No blocking, no trailing gemm --
one V1 CTA kernel and nothing else.** So whether the failures survive nb=32
separates two different bugs with two different fixes.

`nb32.csv`, six repeats per type, n=1024, batch=256, both calls native
(`info_nonzero` out of 256):

| type | nb=32 (one V1 solve) | nb=64 (blocked) |
|---|---|---|
| float | 7, 1, 2, 10, 3, 2 | 80, 86, 83, 78, 71, 73 |
| double | 4, 3, 1, 1, 0, 0 | 12, 12, 21, 12, 7, 10 |
| cfloat | 2, 0, 0, 0, 0, 1 | 27, 31, 28, 22, 23, 18 |
| cdouble | 15, 3, 7, 6, 6, 5 | 58, 56, 46 |

**It survives, for all four types.** The blocking above it makes the failure
10x more likely (float 1-10 against 71-86) but is not the cause: with the
blocking removed entirely, float still loses up to 10 items in 256 and cdouble
up to 15.

The control, `nb32ctl.csv` -- the same nb=32, the gemm STILL native, only the
trsm moved to the vendor, four repeats per type:

| type | nV: native gemm + vendor trsm, nb=32 |
|---|---|
| float | 0, 0, 0, 0 |
| double | 0, 0, 0, 0 |
| cfloat | 0, 0, 0, 0 |
| cdouble | 0, 0, 0, 0 |

Sixteen clean runs. The native gemm is not contributing at nb=32 either.

So the defect is in `trsm_native_v1_dispatch` / `TrsmCtaKernel<T,32,Side::Right>`
itself, not in `trsm_native_blocked`'s decomposition. That is consistent with the
32-aligned `info` distribution of section 3 and it is the single most useful
thing this benchmark found for whoever fixes it.

nb=32 is not a mitigation in any case: against the shipped nb it costs 2.37x
(float, 40.7 against 17.2 ms), 1.09x (double), 1.19x (cfloat) and 1.04x
(cdouble) -- and still returns wrong answers.

## 8. What this means for Phase 3

Routing is Phase 3's job and `preferred()` was NOT touched here. These are the
measurements Phase 3 has to work from, in the order they matter.

1. **`preferred()` must stay all-false for potrf in a vendor-present build.**
   There is no window in this grid where the blocked driver wins on a correct
   answer: 40 cells x 3 configurations, best reliably-correct ratio 0.996x,
   geomean 0.74 in the most favourable configuration. Flipping any part of it
   costs the library speed on every shape it touches.
   The vendor-free build reaches the driver through `route_resolve.hh:60-63`
   regardless, which is the whole point and needs no `preferred()` change.

2. **The native trsm defect is a correctness blocker for vendor-free potrf and
   outranks every performance item here.** A vendor-free build today returns
   wrong Cholesky factors at batch >= 64..256. Nothing in Phase 3 should ship
   while that is true. What this benchmark can hand the fixer:

   * it is the panel trsm and nothing else -- `nV` (native gemm, vendor trsm) is
     clean in every cell tried, at the shipped nb and at nb=32 (section 3,
     section 7);
   * it is in the V1 CTA kernel, not in `trsm_native_blocked`'s decomposition:
     nb=32 reduces the panel solve to ONE `trsm_native_v1_dispatch` call with no
     blocking and no gemm, and it still fails for all four types (section 7);
   * the failing column is 32-ALIGNED -- 69% of 2070 recorded `info` values land
     exactly on a multiple of 32 and 93% within 3 of one, against a 12.5%
     uniform null (section 3);
   * it needs batch. Clean at batch <= 32 in every cell measured, and the rate
     grows to 85 of 256 items (section 3);
   * `nb` is NOT a usable knob for it: every legal value fails, and nb=32 costs
     2.37x for float while still failing.

   The one mitigation that does work is routing the panel solve to the vendor,
   which is exactly what a vendor-free build cannot do.

3. **A register-tiled complex GEMM is the largest single performance item.** It
   is worth 2.7x on vendor-free cdouble potrf by itself (0.32x -> 0.89x of
   cuSOLVER by arithmetic on the measured split), and it is the reason complex
   gets WORSE with n while the real types converge to parity. It is not a potrf
   change at all: `route_gemm.hh:113-114` and `gemm_kernels.cc:471,:728`.

4. **Double is essentially at parity vendor-free from n >= 512** -- 0.98-1.00 at
   n=512, 1.006-1.013 at n=1024, 0.976-0.982 at n=2048, geomean 0.83 over the
   whole double column against 0.36 for cdouble -- and float reaches 0.81-1.02
   at n=1024. For those two types vendor freedom is nearly free at large order
   and expensive only below n ~ 512. (The n=1024 double cells are >1.00 and are
   also WRONG; their correct counterpart `VV` is 0.79, so "parity" here means
   0.79-1.01 depending on what a corrected panel solve costs.)

5. **The small-n deficit is Phase 1's, not Phase 2's.** At n=128 float the CTA
   leaf is 0.293 ms against cuSOLVER's 0.140 (0.48x) and the blocked driver is
   the same leaf plus one launch. No blocked-driver tuning addresses it.

6. **A gemm routing gate to look at, found incidentally.** At batch 32 the
   trailing update goes to cuBLAS because `route_gemm.hh:122`'s `s.batch < 64`
   gate blocks the native kernel, and the native kernel is FASTER there:
   double n=2048 batch=32 measures `def` (vendor gemm) 107.2 ms against `nn`
   (native) 80.3 ms, reproduced three times in `recheck.csv`; float the same
   cell 22.5 against 18.9. That gate is a shape-level `preferred()` question for
   gemm, not for potrf, and it is consistent with the recorded finding that
   cuBLAS is weak at small batched DGEMM.

## 9. Where the earlier phases' numbers did not transfer

Recorded because each was used as an input to a design decision.

1. **"Native/vendor on the trailing shapes is 0.13-0.18x for float"**
   (`potrf_native.hh:167-170`, `factorization.cc:244-246`). Measured on the
   driver's real trailing update it is **1.03x -- native is 3% FASTER** (nsys,
   section 6). The measure phase timed a single `m2 x m2 x nb` gemm; the driver
   issues 217 gemms of `32 x 32 x 128` and `mr x 32 x 128` because of the
   triangular decomposition. The conclusion drawn from the old figure -- inject
   the gemm rather than hardcode it -- is still right, and is now right for a
   better reason: the router picks per call and the shapes differ per call.

2. **"The nb round-down to a multiple of `trsm_cta_max_n<T>()` contains the trsm
   defect"** (`potrf_blocked.cc`). It contains the deterministic mode only. The
   sporadic mode the same measure-phase note recorded for orders 128 and 160
   survives at every legal nb and scales with batch to 85 failures in 256 items
   (section 3). The mitigation is not wrong, it is insufficient.

3. **"W is monotonic above 32 for three types and above 16 for cdouble, on a
   native gemm"** (`potrf_blocked.cc:137-144`), measured on the STAGED driver at
   n=512, batch 128. On the real driver at n=1024, batch 256 the minimum sits
   at: float **128** (not 32, and by 6.5%), double **16** (not 32, but by only
   1.3%), cfloat 32 (reproduced), cdouble 16 (reproduced). float's curve is not
   monotonic at all -- 27.82 / 17.27 / 18.00 / 16.21 / 19.55 for
   W = 16/32/64/128/256 -- which a wasted-work argument cannot produce and a
   kernel-variant boundary can. See section 7.

4. **`BATCHLAS_KERNEL_TRACE` is not a route oracle in this build** -- already
   recorded by the implementer, confirmed here, and the reason attribution is by
   nsys. It emits `sycl_submit` / `sycl_parallel_for` with no kernel names.

## 10. Files

| file | what |
|---|---|
| `bench.cpp`, `build.sh` | the harness. `build.sh` -> `./bench` (vendor build), `build.sh novendor` -> `./bench_nv`. Modes `route`, `ab`, `facade`. |
| `run_route.sh` -> `route.txt` | the route pin, asked of the resolver (section 2). |
| `run_main.sh` -> `main.csv` | the main grid. `analyse.py main.csv` -> `summary_main.txt`. |
| `run_recheck.sh` -> `recheck.csv` | the discarded and suspect cells, 7 reps x 3 passes. |
| `make_overrides.py` -> `overrides.csv` | folds `recheck.csv` and `wins.csv` into the per-cell replacements `analyse.py` marks with `+`. |
| `run_wins.sh` -> `wins.csv` | every cell that read >= 1.00, 9 reps x 3 passes (section 5). |
| `run_novendor.sh` -> `novendor.csv` | the same grid in `build-novendor`, no env (section 4). |
| `run_bisect.sh` -> `bisect.csv` | which injected call causes the failures, first pass. |
| `run_allvendor.sh` -> `allvendor.csv` | the same question with both calls forced, 4 configurations (section 3). |
| `run_batchdep.sh` -> `batchdep.csv` | batch dependence of the failures, through the facade. |
| `run_nb.sh` -> `nbsweep.csv` | nb and W sweeps (section 7). |
| `run_nb32.sh` -> `nb32.csv` | whether nb=32 dodges the trsm defect (section 7). |
| `run_nb32ctl.sh` -> `nb32ctl.csv` | its control: nb=32 with the gemm still native and the trsm on the vendor. |
| `run_nsys.sh`, `nsys_summary.py` -> `nsys/split.txt` | stage attribution (section 6). |
| `gpu_guard.sh` | the standard guard, copied from `experiments/`. |
| `.gitignore` | excludes the two compiled harnesses and the 80 MB of raw nsys captures. The per-kernel CSVs they were reduced to, and `nsys/split.txt`, ARE tracked. |

Every `run_*.sh` is self-contained, takes `GPU=` (default 1) and writes one CSV.
`*.err` beside each CSV holds the `INFONZ` lines naming which batch items failed,
with what `info` value, and the residual of the first failing item.

### Re-running

```
bash build.sh                     # ./bench      against build/
bash build.sh novendor            # ./bench_nv   against build-novendor/
bash run_route.sh                 # verify the pin FIRST, always
bash run_main.sh                  # ~50 min
python3 analyse.py main.csv
```

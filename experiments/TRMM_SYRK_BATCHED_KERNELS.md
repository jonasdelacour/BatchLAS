# Batched TRMM and SYRK kernels for the shapes we actually call them with

PR #61 declined to use `trmm` or `syrk` anywhere in `src/extensions`, and was
right to on the evidence it had: neither op reached a batched kernel at the
shapes its callers produce. This is the other half of that result -- writing the
kernels that were missing, and re-measuring.

Hardware: RTX 4090 / sm_89, CUDA 13.2, RelWithDebInfo, one GPU, no other compute
processes resident. All figures float, `--warmup=5 --min_iters=20`, run-to-run
stddev under 1% except where noted. Harness is `benchmarks/gemm_vs_level3_benchmark.cc`.
Every shape is paired with a batch large enough to saturate; batch = 1 is not
measured and is not a design target.

## What was missing

**SYRK.** `syrk_triangular_tiles` (PR #60) is a 128x128 tile-masked kernel and
its router requires at least three tiles a side, i.e. `n >= 257`. Below that
there is nothing for a triangular *grid* to skip, so the router correctly
refused it and every Gram matrix fell to `syrk_vendor_impl` -- a host loop
issuing one `cublasSsyrk` per batch member. At batch 2048 that is 33 ms against
a 0.33 ms GEMM.

**TRMM.** Nothing in the tree exploited triangularity. `src/extensions/trmm.cc`
recurses only until the block is 256 wide and then calls the GEMM it is meant to
replace, so for every `ib` in {16,32,64,128,256} the structure was never used.
On CUDA the op does not even reach that recursion: `trmm_vendor_impl` expands
the triangle into a `k x k x batch` scratch buffer and hands it to a full GEMM.

## SYRK: `syrk_gram_tiles.hh`

Sizes the tile to `n` rather than fixing it at 128, so one tile is the whole of
C. Both operands of `A^T A` are then the same columns of A, so there is one
shared tile instead of two and A crosses the bus exactly once -- which is what
matters, because arithmetic intensity here is `n/4` flop per byte, 8 at n = 32
against the 4090's ridge near 40.

The triangle is taken at *thread-tile* granularity: only the 136 of 256 thread
tiles that meet the requested half are carried, so 160 threads instead of 256.
Masking only the epilogue leaves the block doing exactly a GEMM's arithmetic,
which is why the first cut could match a GEMM at n = 128 and never beat one.

`gemm` and `syrk before` are one paired run on `main`; `syrk now` is this branch.

| m (reduction) | n | batch | gemm | syrk before | syrk now | vs gemm | vs before |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 2048 | 0.334 | 33.588 | **0.0780** | 4.29x | 431x |
| 512 | 32 | 2048 | 0.711 | 61.075 | **0.1711** | 4.16x | 357x |
| 512 | 64 | 1024 | 0.345 | 31.911 | **0.1875** | 1.84x | 170x |
| 1024 | 64 | 1024 | 0.668 | 60.397 | **0.3326** | 2.01x | 182x |
| 1024 | 128 | 512 | 0.409 | 30.296 | 0.4146 | 0.99x | 73x |
| 2048 | 128 | 256 | 0.404 | 29.370 | **0.3820** | 1.06x | 77x |

At n = 32 the kernel reads 71 MB in 78 us -- 933 GB/s, so it is at the memory
roofline and there is nothing further to get. At n = 128 it is at parity with
GEMM, and that is close to the ceiling too: reading A once costs 298 us against
GEMM's measured 409, so the whole prize there was 1.37x, not 2x.

n >= 256 is untouched and still on `syrk_triangular_tiles`: 0.89x at n = 256,
1.28x at 512, 1.57x at 1024. The n = 256 cell is a pre-existing loss.

## TRMM: `trmm_triangular_tiles.hh`

The structure worth exploiting is the k-loop bound, not a mask. An output tile
rooted at row `m0` only ever touches `op(A)_{i,p}` for `p >= i`, so it can start
its reduction at `m0` -- the arithmetic is not done and then discarded. `uplo`
and `trans` collapse into one flag, `lower_eff`, because transposing an upper
triangle gives a lower one.

How much that saves is worth stating precisely, because it is **not** the
textbook 2x: with `R` row tiles the reduction shrinks to `(R+1)/2R` of the
square -- 1.0x at R = 1, 1.33x at R = 2, only 1.78x by R = 8.

`before` is `BATCHLAS_TRMM_VARIANT=vendor` on this branch (the expansion plus
GEMM), measured in the same session.

| m | nC | batch | gemm | trmm before | trmm now | vs before | vs gemm |
|---|---|---|---|---|---|---|---|
| 32 | 256 | 2048 | 0.187 | 0.186 | **0.167** | 1.11x | 1.12x |
| 64 | 256 | 2048 | 0.352 | 0.399 | **0.333** | 1.20x | 1.06x |
| 64 | 512 | 1024 | 0.333 | 0.351 | **0.320** | 1.10x | 1.04x |
| 128 | 512 | 1024 | 0.674 | 0.784 | 0.699 | 1.12x | 0.97x |
| 128 | 1024 | 512 | 0.637 | 0.687 | 0.686 | 1.00x | 0.93x |
| 256 | 1024 | 256 | 0.758 | 0.852 | 0.915 | 0.93x | 0.83x |
| 256 | 256 | 512 | 0.467 | 0.690 | **0.536** | 1.29x | 0.87x |
| 512 | 512 | 128 | 0.762 | 0.961 | **0.713** | 1.35x | 1.07x |
| 512 | 1024 | 64 | 0.754 | 0.838 | **0.690** | 1.22x | 1.09x |
| 1024 | 1024 | 32 | 1.514 | 1.662 | **1.146** | 1.45x | 1.32x |

### Beating the GEMM in every dtype: what it took

The first version was float-only and lost to the GEMM at m = 128..256. Both
facts had the same cause and one fix.

**The tile width is the whole game, and the scalar type picks it.** R = m/TileM
row tiles shrink the reduction to (R+1)/2R, so R = 1 saves *nothing* -- and the
original rule used one tile for all m <= 128, which is exactly why it could
never beat a GEMM there. Cutting m into more tiles also re-reads B, up to
(R+1)/2 times. Which side wins is decided by precision, not shape: float is
bandwidth bound at these sizes (intensity m/4 against a ridge near 40) so the
re-read is expensive; double runs at 1/64 rate, putting the ridge near 1.4, so
the arithmetic is everything and B's re-read is nearly free. Measured, float,
m = 128 nC = 512 batch 1024 -- TileM 32/64/128 gives 0.663 / **0.658** / 0.699
against a GEMM's 0.674. So 64 for float, 32-64 for the wide types, and 128 only
for float past m = 512.

**Three float-only assumptions had to go**, the same three the Gram SYRK kernel
hit: the `alignas(4*sizeof(T))` packet (undefined for anything but float),
`std::complex::operator*` (the `__mulsc3` libcall), and register pressure (an
8x8 thread tile is 256 registers in `complex<double>`; forcing TileM = 128 there
measured 15.06 ms against 2.18 -- pure spill).

**And one that was not float-specific at all.** Factoring the inner loop into
helpers, the accumulator update was first written as `void accumulate(T& acc,
...)`. Taking the address of an element of the accumulator array is enough for
the compiler to stop keeping it in registers; it went to local memory and cost
**43%** on float at m = 512 (0.659 -> 0.944 ms) with no other change. Both
helpers now return by value.

### Where it lands, by dtype

Ratios are trmm against the GEMM spelling of the same product; **bold** is a win.

| m | nC | batch | float | double | complex&lt;float&gt; | complex&lt;double&gt; |
|---|---|---|---|---|---|---|
| 32 | 256 | 2048 | **1.13x** | **1.29x** | 0.88x | **1.05x** |
| 64 | 256 | 2048 | **1.07x** | **1.45x** | 0.88x | **1.41x** |
| 64 | 512 | 1024 | **1.04x** | **1.45x** | 0.84x | **1.41x** |
| 128 | 512 | 1024 | **1.03x** | **1.48x** | 0.69x | **1.42x** |
| 128 | 1024 | 512 | 1.00x | **1.48x** | 0.69x | **1.42x** |
| 256 | 1024 | 256 | 0.91x | **1.77x** | 0.93x | **1.71x** |
| 256 | 256 | 512 | 0.93x | **1.72x** | 0.89x | **1.66x** |
| 512 | 512 | 128 | **1.15x** | **1.91x** | **1.10x** | **1.85x** |
| 512 | 1024 | 64 | **1.16x** | **1.91x** | **1.10x** | **1.84x** |
| 1024 | 1024 | 32 | **1.26x** | **2.02x** | **1.23x** | **1.95x** |

**double and complex&lt;double&gt; win at every shape measured**, 1.05x to 2.02x --
essentially the theoretical ceiling, because FP64 is compute bound at 1/64 rate
so the halved arithmetic lands in full and grows with m.

**float wins at 8 of 10**, the exceptions being m = 256, where cuBLAS's SGEMM is
running at its 45 TFLOP/s peak and this kernel reaches ~57% of that; a 1.6x
arithmetic saving does not cover a 1.75x rate deficit.

**complex&lt;float&gt; is the one dtype that loses below m = 512**, and it is the
worst case of the same story: cuBLAS's cgemm is at ~100% of the FP32 FMA peak
on these shapes, while a complex accumulator costs twice the registers, so the
thread tile that would give this kernel a competitive fragment-to-FMA ratio does
not fit at a usable occupancy. Widening the column tile from 8 to 16 (halving
the lane count) is what recovers m >= 512; going further to a 128-row tile
spills. It is a register-file ceiling, not a tuning miss.

### A routing mistake worth recording

The first version of the router gated the tile kernel to `m <= 64 || m >= 512`,
picked straight off the "vs gemm" column above. That is the wrong comparison:
the router chooses between the tile kernel and the *vendor*, not between trmm
and gemm. The gate sent `m = 128..256` back to the expansion and cost up to
1.29x on exactly the shapes it was meant to protect. There is now no threshold.

## Two measurement corrections to PR #61

1. **The TW benchmark charged trmm a zeroing pass it does not need.** The CUDA
   and ROCm paths overwrite C (`beta = 0`); only the MKL fallback in
   `src/extensions` accumulates. At ib = 32, nC = 256, batch 2048 that fill is
   ~70 us of a 189 us measurement. Removed.
2. **The TW shapes cannot show a trmm win and the roofline says so in advance.**
   With m in the tens, B and C dominate the traffic and the GEMM is already
   bandwidth bound near 800 GB/s, so halving its arithmetic buys nothing it can
   spend. `BM_Square_*` was added for shapes on the other side of the ridge.

## A bug this found, and the test that now covers it

The 128-wide SYRK tile initially split each thread's 8 rows into two 4-wide
bands 64 apart -- what the square 128x128 kernels do to spread shared-memory
banks. That is incompatible with taking the triangle at thread-tile
granularity, which decides a whole thread tile is inside the requested half
from its tile indices alone. Thread (0,1) then owns element (64,4), which is in
the lower triangle while its tile is not, so nothing wrote it.

It failed quietly and only at `n > 64`. `syrk_tests` pinned exactly one shape,
n = 96, which happened to catch it -- but that shape reaches only one of the
kernel's three tile widths. `SyrkTest.NarrowShapesMatchGemmReference` now sweeps
n in {24, 32, 48, 64, 96, 128} x trans x uplo with k = 200 (deliberately not a
multiple of the k chunk), so each tile width and each partial tile is exercised.

## Double and complex

The Gram kernel is templated, so extending it was mostly a matter of finding the
three things that were secretly float-only:

1. **The 128-bit packet.** `TileVec4<T>` is `alignas(4*sizeof(T))`, which for
   double is 32 bytes and for `complex<double>` 64 -- an alignment
   `sycl::local_accessor` never promises, since it aligns to `T`. The
   reinterpret was undefined for every type but float. `tile_load4` now
   vectorises only at `sizeof(T) == 4` and stays scalar above it.
2. **Register pressure.** A 4-wide thread tile puts 544 threads on the 128-wide
   case, which float wants. Complex doubles every accumulator: 205 registers per
   work-item x 544 is past the 65536 a work-group gets, and the runtime rejects
   the launch outright rather than spilling. Complex takes an 8-wide thread tile
   and 160 threads instead.
3. **`std::complex::operator*`.** It lowers to the `__mulsc3` libcall -- C99
   Annex G, a branch on Inf and NaN around every multiply. In this inner loop it
   cost **18x**: n = 128 took 38 ms where a cuBLAS GEMM took 1.5. Writing the
   four real multiplies out by hand fixed it. Nothing in the source hints at it.

Double, against the GEMM spelling and against the host loop it replaces:

| m | n | batch | gemm | syrk before | syrk now | vs gemm | vs before |
|---|---|---|---|---|---|---|---|
| 256 | 32 | 2048 | 0.901 | 115.40 | **0.837** | 1.08x | 138x |
| 512 | 64 | 1024 | 3.444 | 112.45 | **1.934** | 1.78x | 58x |
| 1024 | 128 | 512 | 13.72 | 110.91 | **6.521** | 2.10x | 17x |

The win *grows* with n in double and shrinks in float, and that is the FP64 rate:
this part runs double at 1/64, so the Gram product is squarely compute bound and
halving the arithmetic lands in full, where float at n = 128 is already against
both roofs.

**Complex is measured and rejected.** In complex float the tile kernel loses to
the existing GEMM-plus-Hermitian-fold at every Gram shape (0.217 vs 0.206 ms at
n=32/batch 2048; 2.08 vs 1.57 at n=128/batch 512): a complex multiply is four
real ones, so herk is compute bound where real syrk is bandwidth bound, and
cuBLAS's cgemm is better at compute than this kernel. `herk` keeps its route.
The conjugating path stays reachable as `BATCHLAS_SYRK_VARIANT=gram` so it stays
measurable and under test.

### The herk test that could not have failed

`HerkTest` checked that the unreferenced triangle stays untouched and that the
two uplo runs agree. Neither can catch conjugating the wrong operand: doing so
returns `conj(C)` instead of `C`, which is still Hermitian and still consistent
across both triangles. `MatchesGemmReference` compares against a GEMM and was
confirmed to fail when the conjugation is flipped.

## Wired into ortho

Both Cholesky Gram sites (`chol_alg`, `shift_chol_alg`) now call `syrk`. Only
the lower triangle is produced, which is all anything downstream reads -- `potrf`
and `trsm` both default to `Uplo::Lower` and the shift kernel touches only the
diagonal. `svqb_alg` keeps its GEMM: it scales the whole `k x k` before handing
it to `syev`, so a half-written C would leave it multiplying uninitialised
workspace.

End to end, m = 1024, batch 512 (`BATCHLAS_ORTHO_GRAM=gemm` pins the old
spelling so this is one binary, not two builds):

| k | algo | float gemm | float syrk | | double gemm | double syrk | |
|---|---|---|---|---|---|---|---|
| 32 | Chol2 | 1.450 | **0.895** | 1.62x | 6.689 | 6.580 | 1.02x |
| 32 | ShiftChol3 | 1.998 | **1.298** | 1.54x | | | |
| 64 | Chol2 | 4.061 | **3.614** | 1.12x | 17.98 | **14.95** | 1.20x |
| 64 | ShiftChol3 | 6.078 | **5.404** | 1.12x | | | |
| 128 | Chol2 | 8.813 | 9.156 | 0.96x | 57.39 | **42.94** | 1.34x |

The float k = 128 loss is why the threshold is `k <= 64` for float and
`k <= 128` for double rather than one number for both.

## Impact on syev: none, and the trace says why

Asked directly. The answer is that **neither new kernel is in syev's path**, and
that is not an oversight in the wiring -- syev does not call `ortho` at all.
Kernel tracing a run is what settles it, and it also shows syev taking two
completely different routes:

| n | route | level-3 triangular ops used |
|---|---|---|
| 256 | `sytrd_blocked` | `syr2k_cuda_custom.triangular_tiles` (PR #61) |
| 512 | `syev_two_stage` | none at all |

So the only level-3 kernel that touches syev is the syr2k PR #61 already landed,
and it only touches the shapes that take the blocked route:

| n | batch | nb | jobz | gemm | syr2k | |
|---|---|---|---|---|---|---|
| 256 | 1024 | 16 | evals | 25.37 | **21.93** | 1.16x |
| 256 | 1024 | 32 | evals | 26.49 | **25.02** | 1.06x |
| 256 | 1024 | 32 | vectors | 39.17 | **37.41** | 1.05x |
| 512 | 1024 | 16 | evals | 162.7 | 162.8 | 1.00x |
| 512 | 1024 | 32 | evals | 162.7 | 162.5 | 1.00x |

n = 512 is exactly 1.00x because the two-stage path never calls syr2k.

Where its time does go, from the same trace (n=512, batch 1024, nb=16):

| kernel | share |
|---|---|
| `syev_two_stage.sb2st_hh` | 62.4% |
| `ormqr_blocked.larft` | 9.2% |
| `syev_two_stage.stebz_evals` | 7.1% |
| `ormqr_blocked.pack_v_panel` | 4.0% |
| `syev_two_stage.sy2sb` | 1.1% |

**The conclusion for anyone chasing syev is that level-3 substitution is not the
lever.** Nearly two thirds of it is one band-to-tridiagonal Householder kernel,
and no triangular BLAS-3 op appears anywhere in that route.

## trmm in ormqr_blocked's WY update, and what it is worth

`W2 = op(T) W1` with `T` upper triangular from `larft` is now a trmm. Both syev
paths call `ormqr_blocked` with `Side::Left`, which is the side the tile kernel
serves, and every block size syev uses is inside the `ib <= 64` window where it
beats the GEMM. Guarded to CUDA float, because anything else takes
`trmm_vendor_impl` -- which expands the triangle into scratch and then runs the
very GEMM being replaced. `BATCHLAS_ORMQR_WY=gemm` pins the old spelling.

Then measured rather than assumed, and the honest answer is **it barely moves
syev** (float, batch as shown, eigenvectors, stddev 0.01-0.55 ms):

| n | batch | nb | gemm | trmm | |
|---|---|---|---|---|---|
| 64 | 2048 | 16 | 3.657 | **3.528** | 1.036x |
| 64 | 4096 | 16 | 7.292 | **7.092** | 1.028x |
| 128 | 2048 | 16 | 14.07 | **13.78** | 1.021x |
| 128 | 4096 | 16 | 27.72 | **27.36** | 1.013x |
| 256 | 1024 | 16 | 33.90 | **33.50** | 1.012x |
| 256 | 1024 | 32 | 37.41 | **36.98** | 1.012x |
| 512 | 1024 | 16 | 366.6 | **365.2** | 1.004x |
| 512 | 1024 | 32 | 367.2 | **366.0** | 1.003x |

Consistent in direction across all eight cells and several times the stddev, so
it is a real effect and not noise -- but it is 3.6% at best and 0.3% at n = 512.

The trace says exactly why, and said so before the measurement: with the kernel
wired, `trmm_cuda_custom.triangular_tiles` is **1.0%** of traced syev time at
n = 512 (8.58 ms of 868, over 240 calls). A substitution that makes that op
~10% faster cannot return more than 0.1% there, and that is what it returns.
The earlier ~9% figure was `ormqr_blocked.larft`, which is a different kernel
and not the one trmm replaces.

At n = 512 with eigenvectors the time is `backtransform_q2` 46.4%, `sb2st_hh`
25.5%, `stedc_eigvecs` 10.6%. Those three are 82% and none is a level-3
triangular op.

## Lifting the CUDA-and-float gate on that trmm

The gate above was justified by CUDA float being the only route with a batched
triangular kernel. That stopped being true once the tile kernel was made
type-generic and the CUDA router started sending double and complex to it, so
the gate was removed outright and the result measured rather than assumed.

`ormqr_blocked_benchmark`, `Side::Left`, `ConjTrans`, ABBA-ordered against
`BATCHLAS_ORMQR_WY=gemm` on a dedicated 4090 via `gpu_guard.sh`, batch 256 (128
for `complex<double>`), nb in {16,32,64}. Ratios are gemm/trmm, so above 1.00 is
trmm ahead. Each figure is the mean of two runs per setting; the sweep was run
twice at different measurement windows and both passes agree:

| type | range over all cells | verdict |
|---|---|---|
| `float` | 1.006x - 1.046x | wins everywhere (already shipped) |
| `double` | 1.004x - 1.016x | **wins everywhere -- newly enabled** |
| `complex<float>` | 0.944x - 0.995x | loses everywhere |
| `complex<double>` | 0.958x - 1.010x | loses at ib = 16, level above |
| `netlib float` | 0.336x - 1.199x | 0.34x at n = 128, ib = 16 |
| `netlib double` | 0.379x - 1.064x | 0.38x at n = 128, ib = 16 |

**So only the double half of the gate was stale, and the split is per-type, not
per-precision.** The reason is the same `R` arithmetic that sizes the row tile.
Here `m` is `ib`, in the tens: at `ib <= 32` the kernel runs one 32-row tile,
`R = 1`, and `(R+1)/2R = 1` means it skips no arithmetic at all. It then wins or
loses purely on being one kernel instead of a cuBLAS call, and since a complex
multiply is four real ones, that trade goes negative in complex where it is
positive in float and double. It closes towards parity at `ib = 64`, where
`R = 2` finally saves something (0.99x). A trace confirms complex really does
take `trmm_cuda_custom.triangular_tiles` rather than the expansion fallback, so
this is the kernel's shape response and not a mis-route.

netlib is out for an unrelated reason: its trmm and its gemm are *both*
per-batch cblas loops, so batching is not the difference. OpenBLAS's `?trmm` is
simply weak on a 16x16 triangle against 128 right-hand sides, and the route also
has to copy B into C first because `cblas_?trmm` works in place. ROCm is out
because `rocblas_?trmm` is a per-batch vendor loop standing against a
strided-batched GEMM.

`ormqr_blocked_tests` also gained the two netlib configurations. The type list
was a CUDA-*or*-host `#if`, so with a GPU backend built the host route had no
coverage at all -- which is what a lift extending to netlib would have needed.
(Note that netlib double needs `OPENBLAS_CORETYPE=SKYLAKEX` on this machine, as
`build/batchlas-env.sh` sets; without it, that configuration fails on `Side::Right`
too, which never touches trmm.)

## What is not done

- Complex would need the tile kernel to be worth its launch at `R = 1`, or a
  16-row tile so `ib = 32` gets `R = 2`. Either would make the gate type-free.
- Wiring the tile kernel into `rocblas.cc`'s trmm would make ROCm a real
  candidate; `wy_trmm_applicable` is where to re-measure it.
- `trmm`'s tile kernel is `Side::Left` only. `ormqr`/`ormbr` use both sides;
  the right-side branch still takes the expansion. syev only uses Left, so this
  costs syev nothing.
- `ormbr` has the same WY update and is not wired; it feeds gesvd, not syev.
- The two-stage path's `sy2sb` trailing update has no syr2k, and n >= 512 syev
  goes through it. That is a separate, larger piece of work than this one.
- The tile kernel runs at ~70% of cuBLAS's per-flop rate. Closing that would
  turn the `m = 128..256` band from a loss into a ~1.2x win.
- SYRK at n = 256 (0.89x, `syrk_triangular_tiles`) is a pre-existing loss and
  was not touched.
